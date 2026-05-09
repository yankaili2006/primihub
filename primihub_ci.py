#!/usr/bin/env python3
"""
PrimiHub CI Test Tool
自动执行 MPC / FL / PSI 测试任务
"""
import json, time, sys, os, subprocess, uuid, argparse
from datetime import datetime

SERVER = "127.0.0.1:50050"
CLI = "./primihub-cli"
API_BASE = "http://127.0.0.1:30811"
GATEWAY_URL = f"{API_BASE}/api"

class PrimiHubCI:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = []
    
    def log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{level}] {msg}")
    
    def run_cli_task(self, name, config_path, timeout=120):
        """通过 CLI 提交任务到 node"""
        self.log(f"Running CLI task [{name}]: {config_path}")
        start = time.time()
        try:
            env = os.environ.copy()
            result = subprocess.run(
                [CLI, "--server", SERVER, "--task_config_file", config_path],
                capture_output=True, text=True, timeout=timeout, env=env
            )
            elapsed = time.time() - start
            
            output = result.stdout + result.stderr
            success = "SUCCESS" in output and "FAILED" not in output
            
            status = "PASS" if success else "FAIL"
            self.results.append((name, status, f"{elapsed:.1f}s"))
            self.log(f"  → {status} ({elapsed:.1f}s)")
            
            if self.verbose or not success:
                for line in output.split("\n"):
                    if "party" in line or "FAILED" in line or "error" in line.lower():
                        print(f"    {line}")
            
            return success
        except subprocess.TimeoutExpired:
            self.results.append((name, "TIMEOUT", f"{timeout}s"))
            self.log(f"  → TIMEOUT (>{timeout}s)")
            return False
        except Exception as e:
            self.results.append((name, "ERROR", str(e)[:30]))
            self.log(f"  → ERROR: {e}")
            return False
    
    def run_api_task(self, name, endpoint, data, headers=None, timeout=60):
        """通过平台 API 提交任务"""
        import requests
        self.log(f"Running API task [{name}]: POST {endpoint}")
        start = time.time()
        try:
            hdrs = {"Content-Type": "application/json", "userId": "1"}
            if headers:
                hdrs.update(headers)
            
            resp = requests.post(
                f"{GATEWAY_URL}/{endpoint}",
                json=data, headers=hdrs, timeout=timeout
            )
            elapsed = time.time() - start
            result = resp.json()
            
            success = result.get("code") == 0
            status = "PASS" if success else "FAIL"
            self.results.append((name, status, f"{elapsed:.1f}s"))
            self.log(f"  → {status} ({elapsed:.1f}s): {result.get('msg', '')}")
            
            if success:
                return result.get("result")
            else:
                if self.verbose:
                    print(f"    {json.dumps(result, ensure_ascii=False)[:300]}")
                return None
        except Exception as e:
            self.results.append((name, "ERROR", str(e)[:30]))
            self.log(f"  → ERROR: {e}")
            return None
    
    def wait_for_fl_task(self, task_id, timeout=600, poll_interval=10):
        """等待 FL 任务完成"""
        import requests
        self.log(f"Waiting for FL task {task_id}...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(
                    f"{GATEWAY_URL}/federatedLearning/getTaskDetails",
                    params={"taskId": task_id},
                    headers={"userId": "1"}, timeout=10
                )
                data = resp.json()
                if data.get("code") == 0:
                    task = data.get("result", {}).get("task", {})
                    state = task.get("taskState")
                    fl = data.get("result", {}).get("federatedLearning", {})
                    if fl:
                        state = fl.get("taskState", state)
                    accuracy = task.get("accuracy", fl.get("accuracy"))
                    loss = task.get("loss", fl.get("loss"))
                    error = task.get("taskErrorMsg", "")
                    
                    if state in (2, "2", "COMPLETED"):
                        self.log(f"  ✅ Completed. accuracy={accuracy}, loss={loss}")
                        return True
                    elif state in (3, "3", "FAILED"):
                        self.log(f"  ❌ Failed: {error}")
                        return False
                    else:
                        elapsed = int(time.time() - start)
                        self.log(f"  ⏳ Running... ({elapsed}s) acc={accuracy}, loss={loss}")
            except Exception as e:
                self.log(f"  ⚠️  Poll error: {e}")
            time.sleep(poll_interval)
        self.log(f"  ⏰ Timeout after {timeout}s")
        return False
    
    def print_summary(self):
        """打印结果汇总"""
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        passed = sum(1 for _, s, _ in self.results if s == "PASS")
        for name, status, detail in self.results:
            icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏰"
            print(f"  {icon} {name:30s} {status:8s} {detail}")
        print("-" * 60)
        print(f"  Total: {len(self.results)}, Passed: {passed}, Failed: {len(self.results)-passed}")
        return passed == len(self.results)


def main():
    parser = argparse.ArgumentParser(description="PrimiHub CI Test Tool")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test", choices=["mpc", "psi", "fl", "all"], default="all")
    parser.add_argument("--server", default="127.0.0.1:50050")
    args = parser.parse_args()
    
    global SERVER
    SERVER = args.server
    
    ci = PrimiHubCI(verbose=args.verbose)
    
    print("=" * 60)
    print("PrimiHub CI Test Tool")
    print(f"Server: {SERVER}")
    print(f"Test mode: {args.test}")
    print("=" * 60)
    
    if args.test in ("mpc", "all"):
        ci.log("=== MPC Tests ===")
        ci.run_cli_task("MPC_ADD", "example/mpc_add_task_conf.json", timeout=30)
        ci.run_cli_task("MPC_SUB", "example/mpc_sub_task_conf.json", timeout=30)
        ci.run_cli_task("MPC_MUL", "example/mpc_mul_task_conf.json", timeout=30)
        ci.run_cli_task("MPC_DIV", "example/mpc_div_task_conf.json", timeout=30)
        ci.run_cli_task("MPC_SUM", "example/mpc_statistics_sum_task_conf.json", timeout=30)
        ci.run_cli_task("MPC_AVG", "example/mpc_statistics_avg_task_conf.json", timeout=30)
        ci.run_cli_task("MPC_MAX", "example/mpc_statistics_max_task_conf.json", timeout=30)
        ci.run_cli_task("MPC_MIN", "example/mpc_statistics_min_task_conf.json", timeout=30)
        ci.run_cli_task("MPC_MissingValue", "example/mpc_missing_value_task_conf.json", timeout=30)
        ci.run_cli_task("MPC_LR", "example/mpc_lr_task_conf.json", timeout=180)
    
    if args.test in ("psi", "all"):
        ci.log("=== PSI Tests ===")
        ci.run_cli_task("PSI_ECDH", "example/psi_ecdh_task_conf.json", timeout=60)
        ci.run_cli_task("PSI_KKRT", "example/psi_kkrt_task_conf.json", timeout=60)
    
    if args.test in ("fl", "all"):
        ci.log("=== FL Tests ===")
        # Submit FL task via platform API
        fl_req = {
            "taskType": 1,
            "algorithmType": 1,
            "federatedType": 1,
            "taskName": f"CI_HFL_lr_{uuid.uuid4().hex[:8]}",
            "projectId": 1,
            "taskDescription": "Auto CI test",
            "isLabelOwner": 1,
            "modelId": None,
            "ownOrganId": "1",
            "ownResourceId": "fl_fake_data",
            "ownFeatures": "x_0,x_1,x_2,x_3,x_4,x_5,x_6",
            "labelFeature": "y",
            "participantOrganIds": "2,3",
            "participantResourceIds": "binclass_hfl_train_client1,binclass_hfl_train_client2",
            "algorithmParams": json.dumps({
                "learning_rate": 1.0, "alpha": 0.0001,
                "batch_size": 100, "global_epoch": 2, "local_epoch": 1,
                "id": "id", "label": "y", "print_metrics": True
            }),
            "modelPath": "data/result/hfl_model.pkl",
            "metricPath": "data/result/hfl_metrics.json"
        }
        result = ci.run_api_task("FL_Create", "federatedLearning/createTask", fl_req)
        if result:
            task_id = result.get("taskId")
            if task_id:
                ci.log(f"FL task ID: {task_id}")
                # FL tasks may need manual dispatch - check after scheduler runs
                ci.log("FL task created. The platform will auto-dispatch via scheduler.")
                ci.log(f"Check progress via: curl '{GATEWAY_URL}/federatedLearning/getTaskDetails?taskId={task_id}'")
    
    return 0 if ci.print_summary() else 1


if __name__ == "__main__":
    sys.exit(main())
