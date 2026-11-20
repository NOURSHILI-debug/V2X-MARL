# H1 Scenario Test Report

## Added tests

`tests/test_highway_scenarios.py` (19 passed): no-scenario baseline equality;
three-profile initialization; 4 agents; 12/20/17 background vehicles; 56-D
observations; `(6,6)` candidate block; 50 m range; zero padding; masks; unchanged
action IDs; seed reproducibility; measured count/gap/speed differences; 300-step
limit; reward formula; invalid selection failures.

## Commands and outcomes

All commands run from repository root, one process, single thread.

```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
python -m pytest tests/test_highway_scenarios.py -q
```
Result: `19 passed in 32.23s`.

```powershell
python -m pytest tests/test_contract.py -v -k highway
```
Result: `5 passed, 4 skipped, 18 deselected in 4.79s`.

```powershell
$env:CONTRACT_STRICT="1"; python -m pytest tests/test_contract.py -v -k highway
```
Result: `9 passed, 18 deselected in 4.94s`.

```powershell
python -m pytest tests/test_reward.py -q
```
Result: `20 passed in 0.09s`.

```powershell
python -m pytest tests/test_highway_adapter.py -q
```
Result: `16 passed in 22.71s`.

```powershell
python -m pytest tests -k "not carla and not toy" -q
```
Result: `192 passed, 4 skipped, 33 deselected in 89.43s`.

Failures: none. Skips are expected strict-mode or unavailable-environment skips.

## Environment

Windows, Python 3.13.5, pytest 9.2.0, highway-env 1.12.1, CPU-only i3 laptop.
No CARLA server was used. `CONTRACT_STRICT=1` was reset before the final suite.

## Preservation and unresolved issues

The committed pre-change baseline confirms unchanged no-scenario observations,
rewards and info (excluding the documented unseeded reset action field). No
read-only listed file was edited. Unresolved: keep-speed gate still passes on all
H1 profiles; relevance definition needs approval; no learner validation was run.

## Batch 1 regression

Scenario-specific tests were extended by `tests/test_scenario_suite.py`.
The seed-reproducibility checks now require identical observations for the same
seed and at least one differing observation for a different seed; they no longer
require every CAV observation to differ and they do not assert on `state()`,
which only contains CAV state and is seed-invariant by design.

Commands and results (one process, single thread):

```powershell
python -m pytest tests/test_scenario_suite.py tests/test_highway_scenarios.py -q
```
Result: `34 passed in 23.68s`.

```powershell
python -m pytest tests/test_contract.py -v -k highway
```
Result: `5 passed, 4 skipped, 18 deselected in 3.89s`.

```powershell
$env:CONTRACT_STRICT="1"; python -m pytest tests/test_contract.py -v -k highway
```
Result: `9 passed, 18 deselected in 7.19s`.

```powershell
python -m pytest tests/test_reward.py tests/test_highway_adapter.py -q
```
Result: `36 passed in 34.94s`.

```powershell
python -m pytest tests -k "not carla and not toy" -q
```
Result: `207 passed, 4 skipped, 33 deselected in 91.92s`.

No read-only file was edited. No CARLA server or learner training was used.
Gate measurements and their G6 failures are in `docs/SCENARIO_GATE_REPORT.md`.
