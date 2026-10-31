import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from bir.resource_ledger import AllocationAttempt, LedgerError, ResourceLedger, ResourceRequest, check_storage, reject_restarted_execution


POLICY = {
    "coordination": {"max_pending_or_running_gpu_jobs": 1, "max_pending_or_running_cpu_only_jobs": 1},
    "gpu": {"total_allocated_gpu_hours": 24, "max_gpu_submissions": 16, "max_gpus_per_job": 4,
            "max_job_minutes": 120, "framework_smoke_minutes": 15, "initial_native_probe_minutes": 60,
            "minimum_release_submission_slots": 1},
    "cpu": {"max_cpus_per_task": 8, "max_ram_gib": 128, "max_cpu_only_job_minutes": 120, "max_cpu_only_core_hours": 32},
    "storage": {"max_tmpdisk_mib": 204800, "max_new_persistent_project_gib": 250,
                "headroom_scope": "remaining_planned_new_project_bytes",
                "minimum_available_to_planned_new_bytes_ratio": 1.20},
}


def gpu_request(**changes):
    return replace(ResourceRequest("gpu", "native_probe", 60, 4, 32, 20480, gpus=4,
                                   justification="Faithful model placement", provenance={"code": "test-fixture"}), **changes)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "private.sqlite"
        self.ledger = ResourceLedger(self.path, POLICY, "project-coordinator")

    def submitted(self, run="probe", job="101", request=None):
        self.ledger.reserve(run, request or gpu_request())
        self.ledger.begin_submission(run)
        self.ledger.attach_job(run, job)

    def test_failures_and_idle_devices_use_actual_allocation_time_once(self):
        self.submitted()
        row = AllocationAttempt("101", 0, 900, 4, 4, "OUT_OF_MEMORY", "1:0")
        self.ledger.reconcile("probe", [row])
        self.ledger.reconcile("probe", [row])
        state = self.ledger.snapshot()
        self.assertEqual(state["used_gpu_hours"], 1)
        self.assertEqual(state["used_cpu_core_hours"], 0)
        self.assertEqual(state["reserved_gpu_hours"], 0)
        self.assertEqual(state["gpu_submissions_and_reservations"], 1)
        with self.assertRaises(LedgerError):
            self.ledger.reconcile("probe", [replace(row, elapsed_seconds=800)])

    def test_unknown_submission_blocks_duplicate_work_and_cannot_cancel_unowned_job(self):
        self.ledger.reserve("probe", gpu_request())
        self.ledger.begin_submission("probe")
        with self.assertRaises(LedgerError):
            self.ledger.reserve("retry", gpu_request())
        with self.assertRaises(LedgerError):
            self.ledger.begin_submission("probe")
        with self.assertRaises(LedgerError):
            self.ledger.assert_owned_job("999")
        self.ledger.attach_job("probe", "101")
        self.assertEqual(self.ledger.assert_owned_job("101"), "probe")
        with self.assertRaises(LedgerError):
            self.ledger.abandon_unsubmitted("probe", "cancel was sent")

    def test_concurrent_reservations_allow_only_one_gpu_allocation(self):
        def reserve(index):
            try:
                ledger = ResourceLedger(self.path, POLICY, "project-coordinator")
                ledger.reserve(f"probe-{index}", gpu_request())
                return True
            except LedgerError:
                return False
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(sum(pool.map(reserve, range(4))), 1)
        self.assertEqual(self.ledger.snapshot()["reserved_gpu_hours"], 4)

    def test_restart_guard_and_complete_reconciliation(self):
        reject_restarted_execution({})
        for raw in ("1", "-1", "unknown"):
            with self.assertRaises(LedgerError):
                reject_restarted_execution({"SLURM_RESTART_COUNT": raw})
        self.submitted()
        self.ledger.mark_restart("probe", 1)
        attempts = [AllocationAttempt("101", 0, 300, 4, 4, "PREEMPTED", "0:15"),
                    AllocationAttempt("101", 1, 10, 4, 4, "FAILED", "1:0")]
        with self.assertRaises(LedgerError):
            self.ledger.reconcile("probe", attempts[:1])
        self.ledger.reconcile("probe", attempts)
        self.assertAlmostEqual(self.ledger.snapshot()["used_gpu_hours"], 4 * 310 / 3600)

    def test_step_rows_and_missing_allocation_are_rejected(self):
        self.ledger.reserve("probe", gpu_request())
        self.ledger.begin_submission("probe")
        with self.assertRaises(LedgerError):
            self.ledger.attach_job("probe", "101.batch")
        self.ledger.attach_job("probe", "101")
        for attempt in (AllocationAttempt("101.batch", 0, 10, 4, 4, "COMPLETED", "0:0"),
                        AllocationAttempt("101", 0, 10, 0, 4, "COMPLETED", "0:0")):
            with self.assertRaises(LedgerError):
                self.ledger.reconcile("probe", [attempt])

    def test_release_reserve_and_caps(self):
        with self.assertRaises(LedgerError):
            self.ledger.reserve("optional", gpu_request(phase="optional"))
        self.ledger.set_release_reserve(22, 0, 1, "pilot-measurement.json")
        with self.assertRaises(LedgerError):
            self.ledger.reserve("optional", gpu_request(phase="optional"))
        self.ledger.reserve("release", gpu_request(phase="release"))
        self.assertEqual(self.ledger.snapshot()["reserved_gpu_hours"], 4)

    def test_cpu_budget_is_separate_and_overspend_is_preserved(self):
        request = ResourceRequest("cpu", "required", 120, 8, 32, 20480, provenance={"code": "fixture"})
        self.submitted(request=request)
        self.ledger.reconcile("probe", [AllocationAttempt("101", 0, 14401, 0, 8, "TIMEOUT", "0:15")])
        self.assertGreater(self.ledger.snapshot()["used_cpu_core_hours"], 32)
        with self.assertRaises(LedgerError):
            self.ledger.reserve("cpu-retry", request)

    def test_coordinator_and_policy_cannot_change_silently(self):
        with self.assertRaises(LedgerError):
            ResourceLedger(self.path, POLICY, "other-agent")
        changed = copy.deepcopy(POLICY)
        changed["gpu"]["total_allocated_gpu_hours"] = 48
        with self.assertRaises(LedgerError):
            ResourceLedger(self.path, changed, "project-coordinator")

    def test_invalid_resource_values_fail_before_reservation(self):
        for changes in ({"minutes": float("nan")}, {"minutes": 61}, {"gpus": 5}, {"gpus": True},
                        {"tmpdisk_mib": 204801}, {"nodes": 2}, {"ram_gib": 129},
                        {"phase": "framework_smoke", "minutes": 16}, {"provenance": None}):
            with self.subTest(changes=changes), self.assertRaises(LedgerError):
                self.ledger.reserve("bad", gpu_request(**changes))

    def test_storage_checks_planned_project_headroom_and_separate_project_cap(self):
        gib = 1024 ** 3
        check = lambda free, current, new: check_storage(POLICY, total_bytes=1000 * gib,
            available_bytes=free * gib, existing_project_bytes=current * gib, planned_new_bytes=new * gib)
        self.assertTrue(check(191, 0, 100)["allowed"])
        self.assertFalse(check(119, 0, 100)["allowed"])
        self.assertFalse(check(600, 200, 51)["allowed"])
        self.assertTrue(check(120, 150, 100)["allowed"])
        self.assertEqual(check(120, 150, 100)["minimum_available_bytes"], 120 * gib)


if __name__ == "__main__":
    unittest.main()
