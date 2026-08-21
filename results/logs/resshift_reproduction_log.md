# ResShift Reproduction Log

## 2026-07-14 - Step 2 start / environment blocker

### Command run

```powershell
$di=[System.IO.DriveInfo]::new('D:\'); Write-Output ('D_FREE_GB={0:N2}' -f ($di.AvailableFreeSpace/1GB)); if ($di.AvailableFreeSpace -lt 6GB) { throw 'DISK_BELOW_6GB' }
```

### stdout/stderr

```text
D_FREE_GB=18.49
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -c "import sys; print(sys.executable); mods=['torch','omegaconf','cv2','numpy','PIL','einops','tqdm','loguru']; import importlib;
for m in mods:
    try:
        mod=importlib.import_module(m); print(m,'OK',getattr(mod,'__version__',''))
    except Exception as e:
        print(m,'ERR',type(e).__name__,str(e)[:200])"
```

### stdout/stderr

```text
D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe
torch OK 2.12.1+cu126
omegaconf ERR ModuleNotFoundError No module named 'omegaconf'
cv2 OK 5.0.0
numpy OK 2.4.4
PIL OK 12.2.0
einops OK 0.8.2
tqdm OK 4.68.4
loguru ERR ModuleNotFoundError No module named 'loguru'
```

### Command run

```powershell
if (Test-Path 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\testdata\RealSet65\0030.jpg') { Get-Item 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\testdata\RealSet65\0030.jpg' | Select-Object FullName,Length,LastWriteTime } else { Write-Output 'TEST_IMAGE_MISSING' }
```

### stdout/stderr

```text
FullName                                                                             Length LastWriteTime
--------                                                                             ------ -------------
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\testdata\RealSet65\0030.jpg  19769 12/3/2023 12:01:34 AM
```

### Command run

```powershell
conda --version
```

### stdout/stderr

```text
conda : The term 'conda' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

### Command run

```powershell
where.exe conda
```

### stdout/stderr

```text
INFO: Could not find files for the given pattern(s).
```

### Command run

```powershell
if (Test-Path 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights') { Get-ChildItem 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights' | Select-Object FullName,Length,LastWriteTime } else { Write-Output 'RESSHIFT_WEIGHTS_DIR_MISSING' }; if (Test-Path 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\weights') { Get-ChildItem 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\weights' | Select-Object FullName,Length,LastWriteTime } else { Write-Output 'BASELINE_WEIGHTS_DIR_MISSING' }
```

### stdout/stderr

```text
RESSHIFT_WEIGHTS_DIR_MISSING
```

### Command run

```powershell
py -0p
```

### stdout/stderr

```text
 -V:3.13 *        C:\Python313\python.exe
```

### Command run

```powershell
where.exe python; where.exe py
```

### stdout/stderr

```text
C:\Python313\python.exe
C:\Windows\py.exe
```

### Command run

```powershell
Get-ChildItem -Path 'C:\','D:\' -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'Python|Miniconda|Anaconda|mambaforge|miniforge' } | Select-Object FullName,LastWriteTime
```

### stdout/stderr

```text
FullName     LastWriteTime
--------     -------------
C:\Python313 2/21/2025 4:53:50 AM
```

### Files read

- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\testdata\RealSet65\0030.jpg` metadata only.
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights` existence check.
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\weights` existence check.

### Files written

- `results/logs/resshift_reproduction_log.md`

### Blocker

The project venv is missing ResShift runtime dependencies `omegaconf` and `loguru`.
The earlier instruction says that if the environment differs from ResShift's pinned stack, create a fresh conda env `resshift` and do not modify the main project env. No `conda` executable is available, and no Python 3.10 interpreter is installed. The only system Python discovered is `C:\Python313\python.exe`, and the verified project venv is Python 3.13 with torch `2.12.1+cu126`.

Stopped before triggering the vendored smoke test because running it would require either modifying the main project venv or using an unavailable separate conda/Python 3.10 environment.

## 2026-07-14 - Option 1 approved / vendored smoke attempt

### Command run

```powershell
if (Test-Path 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\basicsr') { Write-Output 'VENDORED_BASICSR_EXISTS'; Get-ChildItem 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\basicsr' | Select-Object Name,Mode,Length,LastWriteTime } else { Write-Output 'VENDORED_BASICSR_MISSING' }
```

### stdout/stderr

```text
VENDORED_BASICSR_EXISTS

Name        Mode   Length LastWriteTime
----        ----   ------ -------------
data        d-----        7/11/2026 3:02:52 AM
utils       d-----        7/11/2026 3:02:52 AM
__pycache__ d-----        7/11/2026 3:02:51 AM
__init__.py -a---- 93     12/3/2023 12:01:34 AM
```

### Command run

```powershell
rg -n "xformers|memory_efficient_attention" 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift'
```

### stdout/stderr

```text
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\cog.yaml:28:    - "xformers==0.0.20"
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\README.md:28:* Python 3.9.16, Pytorch 1.12.1, [xformers](https://github.com/facebookresearch/xformers) 0.0.20
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\environment.yml:3:  - xformers
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\environment.yml:68:  - xformers=0.0.20=py39_cu11.6.2_pyt1.12.1
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\requirements.txt:11:xformers
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\models\unet.py:22:    import xformers
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\models\unet.py:23:    import xformers.ops as xop
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\models\unet.py:287:            a = xop.memory_efficient_attention(q, k, v, p=0.0)  # b x length x heads x ch
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\models\unet.py:327:            a = xop.memory_efficient_attention(q, k, v, p=0.0)  # b x length x heads x length
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\attention.py:13:    import xformers
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\attention.py:14:    import xformers.ops
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\attention.py:233:        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\attention.py:249:        "softmax-xformers": MemoryEfficientCrossAttention
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\attention.py:254:        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\diffusionmodules\model.py:12:    import xformers
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\diffusionmodules\model.py:13:    import xformers.ops
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\diffusionmodules\model.py:17:    print("No module 'xformers'. Proceeding without it.")
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\diffusionmodules\model.py:207:        Uses xformers efficient implementation,
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\diffusionmodules\model.py:258:        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\diffusionmodules\model.py:281:    assert attn_type in ["vanilla", "vanilla-xformers", "memory-efficient-cross-attn", "linear", "none"], f'attn_type {attn_type} unknown'
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\diffusionmodules\model.py:283:        attn_type = "vanilla-xformers"
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\diffusionmodules\model.py:288:    elif attn_type == "vanilla-xformers":
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -c "import importlib; mods=['omegaconf','loguru','antlr4','lmdb','einops','timm','lpips'];
for m in mods:
    try:
        mod=importlib.import_module(m); print(m,'OK',getattr(mod,'__version__',''))
    except Exception as e:
        print(m,'ERR',type(e).__name__,str(e)[:160])"
```

### stdout/stderr

```text
omegaconf ERR ModuleNotFoundError No module named 'omegaconf'
loguru ERR ModuleNotFoundError No module named 'loguru'
antlr4 ERR ModuleNotFoundError No module named 'antlr4'
lmdb ERR ModuleNotFoundError No module named 'lmdb'
einops OK 0.8.2
timm OK 1.0.27
lpips OK
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -m pip install omegaconf==2.3.0 loguru==0.7.0
```

### stdout/stderr

```text
Collecting omegaconf==2.3.0
  Downloading omegaconf-2.3.0-py3-none-any.whl.metadata (3.9 kB)
Collecting loguru==0.7.0
  Downloading loguru-0.7.0-py3-none-any.whl.metadata (22 kB)
Collecting antlr4-python3-runtime==4.9.* (from omegaconf==2.3.0)
  Downloading antlr4-python3-runtime-4.9.3.tar.gz (117 kB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Requirement already satisfied: PyYAML>=5.1.0 in D:\CSE_21201016_BRAC\CONFERENCE\.venv\Lib\site-packages (from omegaconf==2.3.0) (6.0.3)
Requirement already satisfied: colorama>=0.3.4 in D:\CSE_21201016_BRAC\CONFERENCE\.venv\Lib\site-packages (from loguru==0.7.0) (0.4.6)
Collecting win32-setctime>=1.0.0 (from loguru==0.7.0)
  Downloading win32_setctime-1.2.0-py3-none-any.whl.metadata (2.4 kB)
Downloading omegaconf-2.3.0-py3-none-any.whl (79 kB)
Downloading loguru-0.7.0-py3-none-any.whl (59 kB)
Downloading win32_setctime-1.2.0-py3-none-any.whl (4.1 kB)
Building wheels for collected packages: antlr4-python3-runtime
  Building wheel for antlr4-python3-runtime (pyproject.toml): started
  Building wheel for antlr4-python3-runtime (pyproject.toml): finished with status 'done'
  Created wheel for antlr4-python3-runtime: filename=antlr4_python3_runtime-4.9.3-py3-none-any.whl size=144613 sha256=47af87c1bc7567b701d1e03d63246ae18d3c371aa85faa78c13896fce6a7a723
  Stored in directory: c:\users\capti\appdata\local\pip\cache\wheels\d5\b3\74\a35b66048c9de6631cd74cbc9475e6feb3e69a467983446bd8
Successfully built antlr4-python3-runtime
Installing collected packages: antlr4-python3-runtime, win32-setctime, omegaconf, loguru

Successfully installed antlr4-python3-runtime-4.9.3 loguru-0.7.0 omegaconf-2.3.0 win32-setctime-1.2.0
```

### Command run

```powershell
Get-Content 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\models\unet.py' -TotalCount 45
```

### stdout/stderr

```text
from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .swin_transformer import BasicLayer

try:
    import xformers
    import xformers.ops as xop
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
```

### Command run

```powershell
Get-Content 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\attention.py' -TotalCount 35
```

### stdout/stderr

```text
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -c "import torch, numpy, omegaconf, loguru, antlr4; print('torch', torch.__version__); print('numpy', numpy.__version__); print('cuda', torch.cuda.is_available(), torch.version.cuda); print('omegaconf', omegaconf.__version__); print('loguru', loguru.__version__); print('antlr4 ok')"
```

### stdout/stderr

```text
torch 2.12.1+cu126
numpy 2.4.4
cuda True 12.6
omegaconf 2.3.0
loguru 0.7.0
antlr4 ok
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' inference_resshift.py -i testdata\RealSet65\0030.jpg -o results_smoke --task realsrx4 --scale 4 --steps 15 --chop_size 256
```

### stdout/stderr

```text
Traceback (most recent call last):
  File "D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\inference_resshift.py", line 10, in <module>
    from sampler import ResShiftSampler
  File "D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\sampler.py", line 22, in <module>
    from datapipe.datasets import create_dataset
  File "D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\datapipe\datasets.py", line 10, in <module>
    from albumentations import SmallestMaxSize
ModuleNotFoundError: No module named 'albumentations'
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -m pip install albumentations==1.3.0 qudida==0.0.4 --no-deps
```

### stdout/stderr

```text
Collecting albumentations==1.3.0
  Downloading albumentations-1.3.0-py3-none-any.whl.metadata (34 kB)
Collecting qudida==0.0.4
  Downloading qudida-0.0.4-py3-none-any.whl.metadata (1.5 kB)
Downloading albumentations-1.3.0-py3-none-any.whl (123 kB)
Downloading qudida-0.0.4-py3-none-any.whl (3.5 kB)
Installing collected packages: qudida, albumentations

Successfully installed albumentations-1.3.0 qudida-0.0.4
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' inference_resshift.py -i testdata\RealSet65\0030.jpg -o results_smoke --task realsrx4 --scale 4 --steps 15 --chop_size 256
```

### stdout/stderr

```text
Traceback (most recent call last):
  File "D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\inference_resshift.py", line 10, in <module>
    from sampler import ResShiftSampler
  File "D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\sampler.py", line 22, in <module>
    from datapipe.datasets import create_dataset
  File "D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\datapipe\datasets.py", line 16, in <module>
    from basicsr.data.transforms import augment
  File "D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\basicsr\__init__.py", line 3, in <module>
    from .data import *
  File "D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\basicsr\data\__init__.py", line 22, in <module>
    _dataset_modules = [importlib.import_module(f'basicsr.data.{file_name}') for file_name in dataset_filenames]
                        ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python313\Lib\importlib\__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\basicsr\data\realesrgan_dataset.py", line 17, in <module>
    from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
  File "D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\basicsr\data\degradations.py", line 8, in <module>
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -c "import runpy, sys, types; import torchvision.transforms.functional as F; shim=types.ModuleType('torchvision.transforms.functional_tensor'); shim.rgb_to_grayscale=F.rgb_to_grayscale; sys.modules['torchvision.transforms.functional_tensor']=shim; sys.argv=['inference_resshift.py','-i','testdata\\RealSet65\\0030.jpg','-o','results_smoke','--task','realsrx4','--scale','4','--steps','15','--chop_size','256']; runpy.run_path('inference_resshift.py', run_name='__main__')"
```

### stdout/stderr

```text
Downloading: "https://github.com/zsyOAOA/ResShift/releases/download/v1.0/resshift_realsrx4_s15.pth" to D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights\resshift_realsrx4_s15.pth

Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import runpy, sys, types; import torchvision.transforms.functional as F; shim=types.ModuleType('torchvision.transforms.functional_tensor'); shim.rgb_to_grayscale=F.rgb_to_grayscale; sys.modules['torchvision.transforms.functional_tensor']=shim; sys.argv=['inference_resshift.py','-i','testdata\\RealSet65\\0030.jpg','-o','results_smoke','--task','realsrx4','--scale','4','--steps','15','--chop_size','256']; runpy.run_path('inference_resshift.py', run_name='__main__')
  File "<frozen runpy>", line 287, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "inference_resshift.py", line 111, in <module>
    main()
  File "inference_resshift.py", line 96, in main
    configs, chop_stride, desired_min_size = get_configs(args)
  File "inference_resshift.py", line 55, in get_configs
    load_file_from_url(
  File "D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\basicsr\utils\download_util.py", line 97, in load_file_from_url
    download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
  File "D:\CSE_21201016_BRAC\CONFERENCE\.venv\Lib\site-packages\torch\hub.py", line 749, in download_url_to_file
    with urlopen(req) as u:
  File "C:\Python313\Lib\urllib\request.py", line 189, in urlopen
    return opener.open(url, data, timeout)
  File "C:\Python313\Lib\urllib\request.py", line 495, in open
    response = meth(req, response)
  File "C:\Python313\Lib\urllib\request.py", line 604, in http_response
    response = self.parent.error(
        'http', request, response, code, msg, hdrs)
  File "C:\Python313\Lib\urllib\request.py", line 533, in error
    return self._call_chain(*args)
  File "C:\Python313\Lib\urllib\request.py", line 466, in _call_chain
    result = func(*args)
  File "C:\Python313\Lib\urllib\request.py", line 613, in http_error_default
    raise HTTPError(req.full_url, code, msg, hdrs, fp)
urllib.error.HTTPError: HTTP Error 404: Not Found
```

### Command run

```powershell
if (Test-Path 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights') { Get-ChildItem -Force 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights' | Select-Object FullName,Length,LastWriteTime,Mode } else { Write-Output 'RESSHIFT_WEIGHTS_DIR_MISSING' }
```

### stdout/stderr

```text
<empty; weights directory exists but contains no files>
```

### Command run

```powershell
git status --short --branch
```

### stdout/stderr

```text
## main...origin/main
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -c "import importlib, torch, numpy; mods=['omegaconf','loguru','albumentations','qudida']; print('torch',torch.__version__); print('numpy',numpy.__version__); print('cuda',torch.cuda.is_available(),torch.version.cuda);
for m in mods:
    mod=importlib.import_module(m); print(m,'OK',getattr(mod,'__version__','no __version__'))"
```

### stdout/stderr

```text
torch 2.12.1+cu126
numpy 2.4.4
cuda True 12.6
omegaconf OK 2.3.0
loguru OK 0.7.0
albumentations OK 1.3.0
qudida OK no __version__
```

### Command run

```powershell
if (Test-Path 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\results_smoke') { Get-ChildItem -Recurse 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\results_smoke' | Select-Object FullName,Length,LastWriteTime,Mode } else { Write-Output 'RESULTS_SMOKE_MISSING' }
```

### stdout/stderr

```text
RESULTS_SMOKE_MISSING
```

### Files read

- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\basicsr\`
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\models\unet.py`
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\ldm\modules\attention.py`
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights\`
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\results_smoke\`

### Files written

- Python environment packages installed into `D:\CSE_21201016_BRAC\CONFERENCE\.venv\Lib\site-packages`: `omegaconf`, `loguru`, `antlr4-python3-runtime`, `win32-setctime`, `albumentations`, `qudida`.
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights\` was created by the vendored auto-download path; no weight files were written because the official checkpoint URL returned 404.
- `results/logs/resshift_reproduction_log.md` updated.

### Blocker

The vendored smoke test reached the official GitHub release URL for `resshift_realsrx4_s15.pth`, but the URL returned HTTP 404:

```text
https://github.com/zsyOAOA/ResShift/releases/download/v1.0/resshift_realsrx4_s15.pth
```

The latest instruction said not to fetch weights manually and to trigger auto-download via the vendored smoke test. Stopped here rather than switching to a Hugging Face mirror without explicit confirmation.

## 2026-07-14 - HF mirror approved / permanent shim / standalone smoke pass

### Files written

- `scripts/resshift/_torchvision_compat.py`

### File content summary

The compatibility module installs an in-memory module named `torchvision.transforms.functional_tensor` when that legacy torchvision path is absent. It exposes `rgb_to_grayscale` from the current `torchvision.transforms.functional` API and calls `install()` on import.

### Command run

```powershell
$urls=@(
  'https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_realsrx4_s15.pth',
  'https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_realsrx4_s15_v1.pth',
  'https://huggingface.co/OwlMaster/restorer_files/resolve/main/autoencoder_vq_f4.pth'
)
foreach ($url in $urls) {
  Write-Output "HEAD_URL=$url"
  & curl.exe -L -I --connect-timeout 30 --retry 3 --retry-delay 5 $url 2>&1 | Select-String -Pattern 'HTTP/|content-length|location|x-linked-size|etag|content-disposition'
}
```

### stdout/stderr

```text
HEAD_URL=https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_realsrx4_s15.pth
HTTP/1.1 404 Not Found
Content-Length: 15

HEAD_URL=https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_realsrx4_s15_v1.pth
HTTP/1.1 302 Found
X-Linked-Size: 477905305
X-Linked-ETag: "97eca0d415b2faa90dfe93ee44bd8cf3ee60ee1e560f95b3b3f544d0aabb948b"
HTTP/1.1 200 OK
content-disposition: inline; filename*=UTF-8''resshift_realsrx4_s15_v1.pth; filename="resshift_realsrx4_s15_v1.pth";
content-length: 477905305

HEAD_URL=https://huggingface.co/OwlMaster/restorer_files/resolve/main/autoencoder_vq_f4.pth
HTTP/1.1 302 Found
X-Linked-Size: 221364711
X-Linked-ETag: "e967eaa2c380152811c53e312990d6967a8a957c4f7590f19490ec72110b2889"
HTTP/1.1 200 OK
Content-Length: 221364711
Content-Disposition: inline; filename*=UTF-8''autoencoder_vq_f4.pth; filename="autoencoder_vq_f4.pth";
```

### Command run

```powershell
$repoW='D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights'
$baseW='D:\CSE_21201016_BRAC\CONFERENCE\baselines\weights'
New-Item -ItemType Directory -Force -Path $repoW | Out-Null
New-Item -ItemType Directory -Force -Path $baseW | Out-Null
$modelUrl='https://huggingface.co/OwlMaster/restorer_files/resolve/main/resshift_realsrx4_s15_v1.pth'
$aeUrl='https://huggingface.co/OwlMaster/restorer_files/resolve/main/autoencoder_vq_f4.pth'
$modelOut=Join-Path $repoW 'resshift_realsrx4_s15.pth'
$aeOut=Join-Path $repoW 'autoencoder_vq_f4.pth'
& curl.exe -L --retry 5 --retry-delay 5 -C - --connect-timeout 30 -o $modelOut $modelUrl
& curl.exe -L --retry 5 --retry-delay 5 -C - --connect-timeout 30 -o $aeOut $aeUrl
Copy-Item -Force -LiteralPath $modelOut -Destination (Join-Path $baseW 'resshift_realsrx4_s15.pth')
Copy-Item -Force -LiteralPath $aeOut -Destination (Join-Path $baseW 'autoencoder_vq_f4.pth')
Get-ChildItem -File -LiteralPath $repoW | Select-Object Name,Length,LastWriteTime
Get-ChildItem -File -LiteralPath $baseW | Select-Object Name,Length,LastWriteTime
```

### stdout/stderr

```text
MODEL_SIZE=477905305
AUTOENCODER_SIZE=221364711

REPO_WEIGHTS:
autoencoder_vq_f4.pth     221364711 2026-07-14 03:01:04
resshift_realsrx4_s15.pth 477905305 2026-07-14 03:00:25

BASELINE_WEIGHTS:
autoencoder_vq_f4.pth     221364711 2026-07-14 03:01:04
resshift_realsrx4_s15.pth 477905305 2026-07-14 03:00:25
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -c "import runpy, sys; sys.path.insert(0, r'D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\scripts\resshift'); import _torchvision_compat; sys.argv=['inference_resshift.py','-i','testdata\\RealSet65\\0030.jpg','-o','results_smoke','--task','realsrx4','--scale','4','--steps','15','--chop_size','256']; runpy.run_path('inference_resshift.py', run_name='__main__')"
```

### stdout/stderr

```text
Chopping size/stride: 256/224
Building the diffusion model with length: 15...
No module 'xformers'. Proceeding without it.
Loading Diffusion model from weights\resshift_realsrx4_s15.pth...
Loading AutoEncoder model from weights\autoencoder_vq_f4.pth...
Working with z of shape (1, 3, 64, 64) = 12288 dimensions.
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.00 GiB. GPU 0 has a total capacity of 8.00 GiB of which 4.43 GiB is free.
```

### Command run

```powershell
nvidia-smi
```

### stdout/stderr

```text
NVIDIA GeForce RTX 4060 Laptop GPU
Memory-Usage: 579MiB / 8188MiB
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -c "from pathlib import Path; from PIL import Image; root=Path(r'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\testdata'); ...
for _,w,h,rel in sorted(items)[:30]: print(f'{rel}\t{w}x{h}')"
```

### stdout/stderr

```text
Bicubicx4\lq_matlab\ILSVRC2012_val_00000067.png    64x64
Bicubicx4\lq_matlab\ILSVRC2012_val_00000079.png    64x64
Bicubicx4\lq_matlab\ILSVRC2012_val_00000151.png    64x64
...
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -c "import runpy, sys; sys.path.insert(0, r'D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\scripts\resshift'); import _torchvision_compat; sys.argv=['inference_resshift.py','-i','testdata\\Bicubicx4\\lq_matlab\\ILSVRC2012_val_00000067.png','-o','results_smoke','--task','realsrx4','--scale','4','--steps','15','--chop_size','256']; runpy.run_path('inference_resshift.py', run_name='__main__')"
```

### stdout/stderr

```text
Chopping size/stride: 256/224
Building the diffusion model with length: 15...
No module 'xformers'. Proceeding without it.
Loading Diffusion model from weights\resshift_realsrx4_s15.pth...
Loading AutoEncoder model from weights\autoencoder_vq_f4.pth...
Working with z of shape (1, 3, 64, 64) = 12288 dimensions.
Processing done, enjoy the results in results_smoke
```

### Command run

```powershell
Get-ChildItem -Recurse 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\results_smoke' | Select-Object FullName,Length,LastWriteTime,Mode
```

### stdout/stderr

```text
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\results_smoke\ILSVRC2012_val_00000067.png 136769 2026-07-14 ...
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -c "from pathlib import Path; from PIL import Image; import numpy as np; root=Path(r'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\results_smoke'); ..."
```

### stdout/stderr

```text
D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\results_smoke\ILSVRC2012_val_00000067.png RGB (256, 256) uint8 19 255
```

### Command run

```powershell
Get-FileHash 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights\resshift_realsrx4_s15.pth' -Algorithm SHA256
Get-FileHash 'D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights\autoencoder_vq_f4.pth' -Algorithm SHA256
```

### stdout/stderr

```text
resshift_realsrx4_s15.pth SHA256 97ECA0D415B2FAA90DFE93EE44BD8CF3EE60EE1E560F95B3B3F544D0AABB948B
autoencoder_vq_f4.pth     SHA256 E967EAA2C380152811C53E312990D6967A8A957C4F7590F19490EC72110B2889
```

### Files read

- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\testdata\RealSet65\0030.jpg`
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\testdata\Bicubicx4\lq_matlab\ILSVRC2012_val_00000067.png`
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\results_smoke\ILSVRC2012_val_00000067.png`

### Files written

- `scripts/resshift/_torchvision_compat.py`
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights\resshift_realsrx4_s15.pth`
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights\autoencoder_vq_f4.pth`
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\weights\resshift_realsrx4_s15.pth`
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\weights\autoencoder_vq_f4.pth`
- `D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\results_smoke\ILSVRC2012_val_00000067.png`
- `results/logs/resshift_reproduction_log.md`

### Outcome

The official no-suffix ResShift GitHub/HF path was stale or absent. The HF mirror file `resshift_realsrx4_s15_v1.pth` was downloaded and saved locally as the vendored filename `resshift_realsrx4_s15.pth`, matching the v1 ResShift code path. The standalone vendored smoke test passed on a bundled 64x64 LQ image and produced a 256x256 RGB result.

## HC18 ResShift wrapper run

- limit: 3
- ids: hc18_0960, hc18_0811, hc18_0312
- out_dir: D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored
- chop_size: 256, fallback_chop_size: 128
hc18_0960: native=1024x1024 final=256x256 dtype=uint8 range=0..175 peak_mib=3376.9 seconds=40.89 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0960.png
hc18_0811: native=1024x1024 final=256x256 dtype=uint8 range=2..197 peak_mib=3376.9 seconds=44.90 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0811.png
hc18_0312: native=1024x1024 final=256x256 dtype=uint8 range=2..238 peak_mib=3376.9 seconds=58.02 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0312.png
SANITY_GATE_STOP processed=3 outputs=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -m py_compile scripts\resshift\_torchvision_compat.py scripts\resshift\run_resshift_on_hc18.py
```

### stdout/stderr

```text
<empty; command exited 0>
```

### Command run

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'; & 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' scripts\resshift\run_resshift_on_hc18.py --limit 3
```

### stdout/stderr

```text
Building the diffusion model with length: 15...
No module 'xformers'. Proceeding without it.
Loading Diffusion model from D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights\resshift_realsrx4_s15.pth...
Loading AutoEncoder model from D:\CSE_21201016_BRAC\CONFERENCE\baselines\repos\ResShift\weights\autoencoder_vq_f4.pth...
Working with z of shape (1, 3, 64, 64) = 12288 dimensions.
hc18_0960: native=1024x1024 final=256x256 dtype=uint8 range=0..175 peak_mib=3376.9 seconds=40.89 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0960.png
hc18_0811: native=1024x1024 final=256x256 dtype=uint8 range=2..197 peak_mib=3376.9 seconds=44.90 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0811.png
hc18_0312: native=1024x1024 final=256x256 dtype=uint8 range=2..238 peak_mib=3376.9 seconds=58.02 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0312.png
SANITY_GATE_STOP processed=3 outputs=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored
D:\CSE_21201016_BRAC\CONFERENCE\.venv\Lib\site-packages\timm\models\layers\__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
D:\CSE_21201016_BRAC\CONFERENCE\.venv\Lib\site-packages\torch\functional.py:505: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\aten\src\ATen\native\TensorShape.cpp:4384.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
D:\CSE_21201016_BRAC\CONFERENCE\.venv\Lib\site-packages\torch\nn\modules\module.py:1096: UserWarning: expandable_segments not supported on this platform (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\c10/cuda/CUDAAllocatorConfig.h:40.)
  return self._apply(lambda t: t.cuda(device))
```

### Command run

```powershell
& 'D:\CSE_21201016_BRAC\CONFERENCE\.venv\Scripts\python.exe' -c "from pathlib import Path; from PIL import Image; import numpy as np; root=Path('results/eval/resshift/test/restored'); for p in sorted(root.glob('*.png')): im=Image.open(p); arr=np.array(im); print(p.name, im.mode, im.size, arr.dtype, int(arr.min()), int(arr.max()))"
```

### stdout/stderr

```text
hc18_0312.png L (256, 256) uint8 2 238
hc18_0811.png L (256, 256) uint8 2 197
hc18_0960.png L (256, 256) uint8 0 175
```

### Files written

- `scripts/resshift/run_resshift_on_hc18.py`
- `results/eval/resshift/test/restored/hc18_0960.png`
- `results/eval/resshift/test/restored/hc18_0811.png`
- `results/eval/resshift/test/restored/hc18_0312.png`
- `results/logs/resshift_reproduction_log.md`

## HC18 ResShift wrapper run

- limit: 150
- ids: hc18_0960, hc18_0811, hc18_0312, hc18_0678, hc18_0746, hc18_0916, hc18_0029, hc18_0689, hc18_0578, hc18_0786, hc18_0204, hc18_0539, hc18_0478, hc18_0576, hc18_0798, hc18_0254, hc18_0361, hc18_0310, hc18_0298, hc18_0915, hc18_0522, hc18_0290, hc18_0362, hc18_0420, hc18_0525, hc18_0937, hc18_0239, hc18_0314, hc18_0594, hc18_0323, hc18_0468, hc18_0321, hc18_0542, hc18_0901, hc18_0256, hc18_0235, hc18_0545, hc18_0365, hc18_0266, hc18_0174, hc18_0516, hc18_0039, hc18_0447, hc18_0054, hc18_0715, hc18_0514, hc18_0192, hc18_0909, hc18_0139, hc18_0031, hc18_0213, hc18_0184, hc18_0572, hc18_0913, hc18_0007, hc18_0643, hc18_0825, hc18_0823, hc18_0331, hc18_0059, hc18_0231, hc18_0736, hc18_0895, hc18_0876, hc18_0923, hc18_0595, hc18_0898, hc18_0477, hc18_0010, hc18_0281, hc18_0951, hc18_0136, hc18_0033, hc18_0237, hc18_0848, hc18_0289, hc18_0067, hc18_0352, hc18_0507, hc18_0855, hc18_0696, hc18_0296, hc18_0208, hc18_0799, hc18_0974, hc18_0306, hc18_0723, hc18_0650, hc18_0070, hc18_0935, hc18_0346, hc18_0432, hc18_0260, hc18_0903, hc18_0198, hc18_0824, hc18_0456, hc18_0835, hc18_0009, hc18_0710, hc18_0479, hc18_0882, hc18_0318, hc18_0363, hc18_0465, hc18_0535, hc18_0588, hc18_0382, hc18_0137, hc18_0063, hc18_0528, hc18_0265, hc18_0649, hc18_0551, hc18_0428, hc18_0533, hc18_0351, hc18_0388, hc18_0277, hc18_0065, hc18_0327, hc18_0679, hc18_0215, hc18_0319, hc18_0947, hc18_0462, hc18_0662, hc18_0929, hc18_0078, hc18_0211, hc18_0072, hc18_0086, hc18_0210, hc18_0328, hc18_0081, hc18_0869, hc18_0060, hc18_0906, hc18_0760, hc18_0209, hc18_0307, hc18_0218, hc18_0424, hc18_0247, hc18_0801, hc18_0712, hc18_0292, hc18_0521, hc18_0344, hc18_0880
- out_dir: D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored
- chop_size: 256, fallback_chop_size: 128
hc18_0960: skipped existing D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0960.png
hc18_0811: skipped existing D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0811.png
hc18_0312: skipped existing D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0312.png
sticky downgrade to chop_size=128
hc18_0678: native=1024x1024 final=256x256 dtype=uint8 range=0..226 peak_mib=1921.0 seconds=21.69 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0678.png
hc18_0746: native=1024x1024 final=256x256 dtype=uint8 range=0..236 peak_mib=1800.9 seconds=26.99 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0746.png
hc18_0916: native=1024x1024 final=256x256 dtype=uint8 range=2..202 peak_mib=1800.9 seconds=41.51 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0916.png
hc18_0029: native=1024x1024 final=256x256 dtype=uint8 range=0..237 peak_mib=1800.9 seconds=46.94 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0029.png
hc18_0689: native=1024x1024 final=256x256 dtype=uint8 range=0..217 peak_mib=1800.9 seconds=48.88 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0689.png
hc18_0578: native=1024x1024 final=256x256 dtype=uint8 range=1..235 peak_mib=1800.9 seconds=55.79 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0578.png
hc18_0786: native=1024x1024 final=256x256 dtype=uint8 range=0..243 peak_mib=1800.9 seconds=66.34 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0786.png
hc18_0204: native=1024x1024 final=256x256 dtype=uint8 range=0..225 peak_mib=1800.9 seconds=55.46 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0204.png
hc18_0539: native=1024x1024 final=256x256 dtype=uint8 range=0..241 peak_mib=1800.9 seconds=61.98 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0539.png
hc18_0478: native=1024x1024 final=256x256 dtype=uint8 range=0..213 peak_mib=1800.9 seconds=73.59 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0478.png
hc18_0576: native=1024x1024 final=256x256 dtype=uint8 range=0..225 peak_mib=1800.9 seconds=79.08 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0576.png
hc18_0798: native=1024x1024 final=256x256 dtype=uint8 range=2..245 peak_mib=1800.9 seconds=70.36 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0798.png
hc18_0254: native=1024x1024 final=256x256 dtype=uint8 range=0..198 peak_mib=1800.9 seconds=55.89 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0254.png
hc18_0361: native=1024x1024 final=256x256 dtype=uint8 range=3..243 peak_mib=1800.9 seconds=55.91 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0361.png
hc18_0310: native=1024x1024 final=256x256 dtype=uint8 range=2..209 peak_mib=1800.9 seconds=53.45 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0310.png
hc18_0298: native=1024x1024 final=256x256 dtype=uint8 range=0..239 peak_mib=1800.9 seconds=58.37 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0298.png
hc18_0915: native=1024x1024 final=256x256 dtype=uint8 range=0..174 peak_mib=1800.9 seconds=56.23 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0915.png
hc18_0522: native=1024x1024 final=256x256 dtype=uint8 range=0..247 peak_mib=1800.9 seconds=56.56 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0522.png
hc18_0290: native=1024x1024 final=256x256 dtype=uint8 range=1..226 peak_mib=1800.9 seconds=55.68 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0290.png
hc18_0362: native=1024x1024 final=256x256 dtype=uint8 range=3..237 peak_mib=1800.9 seconds=61.45 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0362.png
hc18_0420: native=1024x1024 final=256x256 dtype=uint8 range=1..196 peak_mib=1800.9 seconds=57.47 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0420.png
hc18_0525: native=1024x1024 final=256x256 dtype=uint8 range=0..169 peak_mib=1800.9 seconds=70.48 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0525.png
hc18_0937: native=1024x1024 final=256x256 dtype=uint8 range=0..185 peak_mib=1800.9 seconds=55.64 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0937.png
hc18_0239: native=1024x1024 final=256x256 dtype=uint8 range=3..239 peak_mib=1800.9 seconds=57.06 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0239.png
hc18_0314: native=1024x1024 final=256x256 dtype=uint8 range=1..234 peak_mib=1800.9 seconds=58.27 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0314.png
hc18_0594: native=1024x1024 final=256x256 dtype=uint8 range=0..251 peak_mib=1800.9 seconds=59.00 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0594.png
hc18_0323: native=1024x1024 final=256x256 dtype=uint8 range=0..227 peak_mib=1800.9 seconds=55.45 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0323.png
hc18_0468: native=1024x1024 final=256x256 dtype=uint8 range=2..238 peak_mib=1800.9 seconds=57.26 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0468.png
hc18_0321: native=1024x1024 final=256x256 dtype=uint8 range=0..216 peak_mib=1800.9 seconds=53.69 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0321.png
hc18_0542: native=1024x1024 final=256x256 dtype=uint8 range=0..228 peak_mib=1800.9 seconds=41.17 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0542.png
hc18_0901: native=1024x1024 final=256x256 dtype=uint8 range=1..159 peak_mib=1800.9 seconds=32.54 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0901.png
hc18_0256: native=1024x1024 final=256x256 dtype=uint8 range=3..251 peak_mib=1800.9 seconds=31.32 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0256.png
hc18_0235: native=1024x1024 final=256x256 dtype=uint8 range=2..212 peak_mib=1800.9 seconds=32.98 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0235.png
hc18_0545: native=1024x1024 final=256x256 dtype=uint8 range=2..248 peak_mib=1800.9 seconds=31.38 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0545.png
hc18_0365: native=1024x1024 final=256x256 dtype=uint8 range=2..172 peak_mib=1800.9 seconds=31.06 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0365.png
hc18_0266: native=1024x1024 final=256x256 dtype=uint8 range=0..232 peak_mib=1800.9 seconds=30.96 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0266.png
hc18_0174: native=1024x1024 final=256x256 dtype=uint8 range=0..219 peak_mib=1800.9 seconds=31.99 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0174.png
hc18_0516: native=1024x1024 final=256x256 dtype=uint8 range=5..220 peak_mib=1800.9 seconds=30.12 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0516.png
hc18_0039: native=1024x1024 final=256x256 dtype=uint8 range=3..253 peak_mib=1800.9 seconds=31.90 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0039.png
hc18_0447: native=1024x1024 final=256x256 dtype=uint8 range=1..245 peak_mib=1800.9 seconds=31.42 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0447.png
hc18_0054: native=1024x1024 final=256x256 dtype=uint8 range=0..227 peak_mib=1800.9 seconds=31.63 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0054.png
hc18_0715: native=1024x1024 final=256x256 dtype=uint8 range=1..241 peak_mib=1800.9 seconds=31.47 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0715.png
hc18_0514: native=1024x1024 final=256x256 dtype=uint8 range=1..243 peak_mib=1800.9 seconds=33.00 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0514.png
hc18_0192: native=1024x1024 final=256x256 dtype=uint8 range=3..248 peak_mib=1800.9 seconds=30.61 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0192.png
hc18_0909: native=1024x1024 final=256x256 dtype=uint8 range=1..239 peak_mib=1800.9 seconds=30.64 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0909.png
hc18_0139: native=1024x1024 final=256x256 dtype=uint8 range=0..209 peak_mib=1800.9 seconds=32.21 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0139.png
hc18_0031: native=1024x1024 final=256x256 dtype=uint8 range=0..219 peak_mib=1800.9 seconds=31.20 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0031.png
hc18_0213: native=1024x1024 final=256x256 dtype=uint8 range=2..242 peak_mib=1800.9 seconds=32.03 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0213.png
hc18_0184: native=1024x1024 final=256x256 dtype=uint8 range=0..254 peak_mib=1800.9 seconds=31.34 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0184.png
hc18_0572: native=1024x1024 final=256x256 dtype=uint8 range=0..227 peak_mib=1800.9 seconds=32.42 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0572.png
hc18_0913: native=1024x1024 final=256x256 dtype=uint8 range=1..208 peak_mib=1800.9 seconds=31.42 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0913.png
hc18_0007: native=1024x1024 final=256x256 dtype=uint8 range=0..227 peak_mib=1800.9 seconds=31.79 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0007.png
hc18_0643: native=1024x1024 final=256x256 dtype=uint8 range=0..239 peak_mib=1800.9 seconds=30.37 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0643.png
hc18_0825: native=1024x1024 final=256x256 dtype=uint8 range=0..236 peak_mib=1800.9 seconds=31.40 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0825.png
hc18_0823: native=1024x1024 final=256x256 dtype=uint8 range=2..198 peak_mib=1800.9 seconds=30.36 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0823.png
hc18_0331: native=1024x1024 final=256x256 dtype=uint8 range=0..244 peak_mib=1800.9 seconds=31.52 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0331.png
hc18_0059: native=1024x1024 final=256x256 dtype=uint8 range=1..228 peak_mib=1800.9 seconds=31.34 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0059.png
hc18_0231: native=1024x1024 final=256x256 dtype=uint8 range=0..196 peak_mib=1800.9 seconds=31.45 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0231.png
hc18_0736: native=1024x1024 final=256x256 dtype=uint8 range=0..246 peak_mib=1800.9 seconds=31.06 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0736.png
hc18_0895: native=1024x1024 final=256x256 dtype=uint8 range=0..174 peak_mib=1800.9 seconds=31.60 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0895.png
hc18_0876: native=1024x1024 final=256x256 dtype=uint8 range=0..148 peak_mib=1800.9 seconds=30.08 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0876.png
hc18_0923: native=1024x1024 final=256x256 dtype=uint8 range=0..180 peak_mib=1800.9 seconds=32.45 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0923.png
hc18_0595: native=1024x1024 final=256x256 dtype=uint8 range=1..222 peak_mib=1800.9 seconds=35.25 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0595.png
hc18_0898: native=1024x1024 final=256x256 dtype=uint8 range=0..234 peak_mib=1800.9 seconds=34.66 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0898.png
hc18_0477: native=1024x1024 final=256x256 dtype=uint8 range=0..194 peak_mib=1800.9 seconds=31.24 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0477.png

## HC18 ResShift wrapper run

- limit: 20
- ids: hc18_0010, hc18_0281, hc18_0951, hc18_0136, hc18_0033, hc18_0237, hc18_0848, hc18_0289, hc18_0067, hc18_0352, hc18_0507, hc18_0855, hc18_0696, hc18_0296, hc18_0208, hc18_0799, hc18_0974, hc18_0306, hc18_0723, hc18_0650
- out_dir: D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored
- chop_size: 256, fallback_chop_size: 128
sticky downgrade to chop_size=128
hc18_0010: native=1024x1024 final=256x256 dtype=uint8 range=0..248 peak_mib=10122.0 seconds=19.15 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0010.png
hc18_0281: native=1024x1024 final=256x256 dtype=uint8 range=1..236 peak_mib=1800.9 seconds=14.06 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0281.png
hc18_0951: native=1024x1024 final=256x256 dtype=uint8 range=0..190 peak_mib=1800.9 seconds=12.80 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0951.png
hc18_0136: native=1024x1024 final=256x256 dtype=uint8 range=1..249 peak_mib=1800.9 seconds=13.61 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0136.png
hc18_0033: native=1024x1024 final=256x256 dtype=uint8 range=0..216 peak_mib=1800.9 seconds=16.54 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0033.png
hc18_0237: native=1024x1024 final=256x256 dtype=uint8 range=5..239 peak_mib=1800.9 seconds=17.68 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0237.png
hc18_0848: native=1024x1024 final=256x256 dtype=uint8 range=0..235 peak_mib=1800.9 seconds=18.05 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0848.png
hc18_0289: native=1024x1024 final=256x256 dtype=uint8 range=0..240 peak_mib=1800.9 seconds=21.13 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0289.png
hc18_0067: native=1024x1024 final=256x256 dtype=uint8 range=2..208 peak_mib=1800.9 seconds=21.82 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0067.png
hc18_0352: native=1024x1024 final=256x256 dtype=uint8 range=0..218 peak_mib=1800.9 seconds=21.61 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0352.png
hc18_0507: native=1024x1024 final=256x256 dtype=uint8 range=0..241 peak_mib=1800.9 seconds=24.19 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0507.png
hc18_0855: native=1024x1024 final=256x256 dtype=uint8 range=0..203 peak_mib=1800.9 seconds=20.11 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0855.png
hc18_0696: native=1024x1024 final=256x256 dtype=uint8 range=1..236 peak_mib=1800.9 seconds=19.38 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0696.png
hc18_0296: native=1024x1024 final=256x256 dtype=uint8 range=0..215 peak_mib=1800.9 seconds=21.46 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0296.png
hc18_0208: native=1024x1024 final=256x256 dtype=uint8 range=0..227 peak_mib=1800.9 seconds=21.09 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0208.png
hc18_0799: native=1024x1024 final=256x256 dtype=uint8 range=0..218 peak_mib=1800.9 seconds=26.55 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0799.png
hc18_0974: native=1024x1024 final=256x256 dtype=uint8 range=0..183 peak_mib=1800.9 seconds=21.16 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0974.png
hc18_0306: native=1024x1024 final=256x256 dtype=uint8 range=0..211 peak_mib=1800.9 seconds=22.03 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0306.png
hc18_0723: native=1024x1024 final=256x256 dtype=uint8 range=0..243 peak_mib=1800.9 seconds=23.89 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0723.png
hc18_0650: native=1024x1024 final=256x256 dtype=uint8 range=0..229 peak_mib=1800.9 seconds=20.83 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0650.png
SANITY_GATE_STOP processed=20 outputs=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored

## HC18 ResShift wrapper run

- limit: 20
- ids: hc18_0070, hc18_0935, hc18_0346, hc18_0432, hc18_0260, hc18_0903, hc18_0198, hc18_0824, hc18_0456, hc18_0835, hc18_0009, hc18_0710, hc18_0479, hc18_0882, hc18_0318, hc18_0363, hc18_0465, hc18_0535, hc18_0588, hc18_0382
- out_dir: D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored
- chop_size: 256, fallback_chop_size: 128
sticky downgrade to chop_size=128
hc18_0070: native=1024x1024 final=256x256 dtype=uint8 range=0..197 peak_mib=10122.0 seconds=14.22 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0070.png
hc18_0935: native=1024x1024 final=256x256 dtype=uint8 range=0..174 peak_mib=1800.9 seconds=11.73 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0935.png
hc18_0346: native=1024x1024 final=256x256 dtype=uint8 range=1..236 peak_mib=1800.9 seconds=11.72 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0346.png
hc18_0432: native=1024x1024 final=256x256 dtype=uint8 range=0..249 peak_mib=1800.9 seconds=12.15 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0432.png
hc18_0260: native=1024x1024 final=256x256 dtype=uint8 range=1..245 peak_mib=1800.9 seconds=13.26 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0260.png
hc18_0903: native=1024x1024 final=256x256 dtype=uint8 range=1..244 peak_mib=1800.9 seconds=15.04 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0903.png
hc18_0198: native=1024x1024 final=256x256 dtype=uint8 range=3..239 peak_mib=1800.9 seconds=17.10 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0198.png
hc18_0824: native=1024x1024 final=256x256 dtype=uint8 range=0..224 peak_mib=1800.9 seconds=18.46 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0824.png
hc18_0456: native=1024x1024 final=256x256 dtype=uint8 range=2..233 peak_mib=1800.9 seconds=18.74 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0456.png
hc18_0835: native=1024x1024 final=256x256 dtype=uint8 range=0..222 peak_mib=1800.9 seconds=19.29 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0835.png
hc18_0009: native=1024x1024 final=256x256 dtype=uint8 range=0..213 peak_mib=1800.9 seconds=19.64 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0009.png
hc18_0710: native=1024x1024 final=256x256 dtype=uint8 range=1..249 peak_mib=1800.9 seconds=21.08 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0710.png
hc18_0479: native=1024x1024 final=256x256 dtype=uint8 range=2..241 peak_mib=1800.9 seconds=24.02 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0479.png
hc18_0882: native=1024x1024 final=256x256 dtype=uint8 range=1..224 peak_mib=1800.9 seconds=20.60 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0882.png
hc18_0318: native=1024x1024 final=256x256 dtype=uint8 range=1..254 peak_mib=1800.9 seconds=22.80 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0318.png
hc18_0363: native=1024x1024 final=256x256 dtype=uint8 range=0..239 peak_mib=1800.9 seconds=21.75 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0363.png
hc18_0465: native=1024x1024 final=256x256 dtype=uint8 range=4..249 peak_mib=1800.9 seconds=22.40 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0465.png
hc18_0535: native=1024x1024 final=256x256 dtype=uint8 range=5..238 peak_mib=1800.9 seconds=22.50 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0535.png
hc18_0588: native=1024x1024 final=256x256 dtype=uint8 range=1..236 peak_mib=1800.9 seconds=21.50 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0588.png
hc18_0382: native=1024x1024 final=256x256 dtype=uint8 range=2..218 peak_mib=1800.9 seconds=22.14 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0382.png
SANITY_GATE_STOP processed=20 outputs=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored

## HC18 ResShift wrapper run

- limit: 20
- ids: hc18_0137, hc18_0063, hc18_0528, hc18_0265, hc18_0649, hc18_0551, hc18_0428, hc18_0533, hc18_0351, hc18_0388, hc18_0277, hc18_0065, hc18_0327, hc18_0679, hc18_0215, hc18_0319, hc18_0947, hc18_0462, hc18_0662, hc18_0929
- out_dir: D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored
- chop_size: 256, fallback_chop_size: 128
sticky downgrade to chop_size=128
hc18_0137: native=1024x1024 final=256x256 dtype=uint8 range=3..251 peak_mib=10122.0 seconds=14.03 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0137.png
hc18_0063: native=1024x1024 final=256x256 dtype=uint8 range=0..230 peak_mib=1800.9 seconds=11.46 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0063.png
hc18_0528: native=1024x1024 final=256x256 dtype=uint8 range=1..198 peak_mib=1800.9 seconds=12.05 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0528.png
hc18_0265: native=1024x1024 final=256x256 dtype=uint8 range=0..211 peak_mib=1800.9 seconds=12.22 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0265.png
hc18_0649: native=1024x1024 final=256x256 dtype=uint8 range=0..233 peak_mib=1800.9 seconds=13.24 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0649.png
hc18_0551: native=1024x1024 final=256x256 dtype=uint8 range=3..239 peak_mib=1800.9 seconds=14.29 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0551.png
hc18_0428: native=1024x1024 final=256x256 dtype=uint8 range=1..235 peak_mib=1800.9 seconds=18.03 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0428.png
hc18_0533: native=1024x1024 final=256x256 dtype=uint8 range=1..235 peak_mib=1800.9 seconds=20.01 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0533.png
hc18_0351: native=1024x1024 final=256x256 dtype=uint8 range=0..233 peak_mib=1800.9 seconds=19.83 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0351.png
hc18_0388: native=1024x1024 final=256x256 dtype=uint8 range=0..229 peak_mib=1800.9 seconds=19.45 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0388.png
hc18_0277: native=1024x1024 final=256x256 dtype=uint8 range=3..224 peak_mib=1800.9 seconds=21.84 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0277.png
hc18_0065: native=1024x1024 final=256x256 dtype=uint8 range=0..240 peak_mib=1800.9 seconds=22.02 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0065.png
hc18_0327: native=1024x1024 final=256x256 dtype=uint8 range=3..225 peak_mib=1800.9 seconds=20.16 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0327.png
hc18_0679: native=1024x1024 final=256x256 dtype=uint8 range=1..227 peak_mib=1800.9 seconds=24.33 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0679.png
hc18_0215: native=1024x1024 final=256x256 dtype=uint8 range=0..231 peak_mib=1800.9 seconds=22.39 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0215.png
hc18_0319: native=1024x1024 final=256x256 dtype=uint8 range=7..247 peak_mib=1800.9 seconds=22.54 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0319.png
hc18_0947: native=1024x1024 final=256x256 dtype=uint8 range=0..181 peak_mib=1800.9 seconds=23.39 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0947.png
hc18_0462: native=1024x1024 final=256x256 dtype=uint8 range=1..226 peak_mib=1800.9 seconds=19.11 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0462.png
hc18_0662: native=1024x1024 final=256x256 dtype=uint8 range=1..226 peak_mib=1800.9 seconds=22.84 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0662.png
hc18_0929: native=1024x1024 final=256x256 dtype=uint8 range=1..148 peak_mib=1800.9 seconds=23.22 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0929.png
SANITY_GATE_STOP processed=20 outputs=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored

## HC18 ResShift wrapper run

- limit: 20
- ids: hc18_0078, hc18_0211, hc18_0072, hc18_0086, hc18_0210, hc18_0328, hc18_0081, hc18_0869, hc18_0060, hc18_0906, hc18_0760, hc18_0209, hc18_0307, hc18_0218, hc18_0424, hc18_0247, hc18_0801, hc18_0712, hc18_0292, hc18_0521
- out_dir: D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored
- chop_size: 256, fallback_chop_size: 128
sticky downgrade to chop_size=128
hc18_0078: native=1024x1024 final=256x256 dtype=uint8 range=0..240 peak_mib=10122.0 seconds=20.24 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0078.png
hc18_0211: native=1024x1024 final=256x256 dtype=uint8 range=2..229 peak_mib=1800.9 seconds=15.89 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0211.png
hc18_0072: native=1024x1024 final=256x256 dtype=uint8 range=1..186 peak_mib=1800.9 seconds=17.72 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0072.png
hc18_0086: native=1024x1024 final=256x256 dtype=uint8 range=0..247 peak_mib=1800.9 seconds=21.38 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0086.png
hc18_0210: native=1024x1024 final=256x256 dtype=uint8 range=0..214 peak_mib=1800.9 seconds=23.24 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0210.png
hc18_0328: native=1024x1024 final=256x256 dtype=uint8 range=2..220 peak_mib=1800.9 seconds=23.14 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0328.png
hc18_0081: native=1024x1024 final=256x256 dtype=uint8 range=2..200 peak_mib=1800.9 seconds=23.27 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0081.png
hc18_0869: native=1024x1024 final=256x256 dtype=uint8 range=2..241 peak_mib=1800.9 seconds=29.10 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0869.png
hc18_0060: native=1024x1024 final=256x256 dtype=uint8 range=1..225 peak_mib=1800.9 seconds=28.14 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0060.png
hc18_0906: native=1024x1024 final=256x256 dtype=uint8 range=0..150 peak_mib=1800.9 seconds=22.99 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0906.png
hc18_0760: native=1024x1024 final=256x256 dtype=uint8 range=0..220 peak_mib=1800.9 seconds=22.83 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0760.png
hc18_0209: native=1024x1024 final=256x256 dtype=uint8 range=0..228 peak_mib=1800.9 seconds=22.17 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0209.png
hc18_0307: native=1024x1024 final=256x256 dtype=uint8 range=0..255 peak_mib=1800.9 seconds=24.29 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0307.png
hc18_0218: native=1024x1024 final=256x256 dtype=uint8 range=0..228 peak_mib=1800.9 seconds=23.16 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0218.png
hc18_0424: native=1024x1024 final=256x256 dtype=uint8 range=3..221 peak_mib=1800.9 seconds=22.17 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0424.png
hc18_0247: native=1024x1024 final=256x256 dtype=uint8 range=3..237 peak_mib=1800.9 seconds=22.91 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0247.png
hc18_0801: native=1024x1024 final=256x256 dtype=uint8 range=0..149 peak_mib=1800.9 seconds=24.86 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0801.png
hc18_0712: native=1024x1024 final=256x256 dtype=uint8 range=0..239 peak_mib=1800.9 seconds=27.17 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0712.png
hc18_0292: native=1024x1024 final=256x256 dtype=uint8 range=4..235 peak_mib=1800.9 seconds=21.27 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0292.png
hc18_0521: native=1024x1024 final=256x256 dtype=uint8 range=0..206 peak_mib=1800.9 seconds=21.32 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0521.png
SANITY_GATE_STOP processed=20 outputs=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored

## HC18 ResShift wrapper run

- limit: 20
- ids: hc18_0344, hc18_0880
- out_dir: D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored
- chop_size: 256, fallback_chop_size: 128
sticky downgrade to chop_size=128
hc18_0344: native=1024x1024 final=256x256 dtype=uint8 range=0..244 peak_mib=10122.0 seconds=16.08 chop=128 fallback=True out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0344.png
hc18_0880: native=1024x1024 final=256x256 dtype=uint8 range=0..169 peak_mib=1800.9 seconds=11.44 chop=128 fallback=False out=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored\hc18_0880.png
SANITY_GATE_STOP processed=2 outputs=D:\CSE_21201016_BRAC\CONFERENCE\Fetal_SR_Project\results\eval\resshift\test\restored


## Step 7: scripts/42_compare_all_models.py
U s i n g   d e v i c e :   c u d a  
 L o a d i n g   p r e t r a i n e d   m o d e l   L P I P S   f r o m   C : \ U s e r s \ c a p t i \ . c a c h e \ t o r c h \ h u b \ p y i q a \ L P I P S _ v 0 . 1 _ a l e x - d f 7 3 2 8 5 e . p t h  
 L o a d i n g   p r e t r a i n e d   m o d e l   D I S T S   f r o m   C : \ U s e r s \ c a p t i \ . c a c h e \ t o r c h \ h u b \ p y i q a \ D I S T S _ w e i g h t s - f 5 e 6 5 c 9 6 . p t h  
 E v a l u a t i n g :   L Q _ m i x e d   b a s e l i n e  
 E v a l u a t i n g :   S R - o n l y  
 E v a l u a t i n g :   Q u a l i t y - a w a r e   S R  
 E v a l u a t i n g :   S R - o n l y   +   L P I P S   +   G A N  
 E v a l u a t i n g :   Q u a l i t y - a w a r e   S R   +   L P I P S   +   G A N  
 E v a l u a t i n g :   S w i n I R - L i g h t   P h a s e   A  
 E v a l u a t i n g :   R e s S h i f t  
 P e r - i m a g e   C S V :   D : \ C S E _ 2 1 2 0 1 0 1 6 _ B R A C \ C O N F E R E N C E \ F e t a l _ S R _ P r o j e c t \ r e s u l t s \ f i n a l _ a l l _ h c 1 8 _ c o m p a r i s o n _ p e r _ i m a g e . c s v  
 S u m m a r y   C S V :   D : \ C S E _ 2 1 2 0 1 0 1 6 _ B R A C \ C O N F E R E N C E \ F e t a l _ S R _ P r o j e c t \ r e s u l t s \ f i n a l _ a l l _ h c 1 8 _ c o m p a r i s o n _ s u m m a r y . c s v  
 M a r k d o w n   t a b l e :   D : \ C S E _ 2 1 2 0 1 0 1 6 _ B R A C \ C O N F E R E N C E \ F e t a l _ S R _ P r o j e c t \ r e s u l t s \ f i n a l _ a l l _ h c 1 8 _ c o m p a r i s o n _ t a b l e . m d  
 S w i n I R   r e c o n   p e r - i m a g e   C S V :   D : \ C S E _ 2 1 2 0 1 0 1 6 _ B R A C \ C O N F E R E N C E \ F e t a l _ S R _ P r o j e c t \ r e s u l t s \ e v a l \ s w i n i r _ l i g h t _ p h a s e _ a \ t e s t \ r e c o n _ p e r _ i m a g e . c s v  
 S w i n I R   r e c o n   s u m m a r y   C S V :   D : \ C S E _ 2 1 2 0 1 0 1 6 _ B R A C \ C O N F E R E N C E \ F e t a l _ S R _ P r o j e c t \ r e s u l t s \ e v a l \ s w i n i r _ l i g h t _ p h a s e _ a \ t e s t \ r e c o n _ s u m m a r y . c s v  
 R e s S h i f t   r e c o n   p e r - i m a g e   C S V :   D : \ C S E _ 2 1 2 0 1 0 1 6 _ B R A C \ C O N F E R E N C E \ F e t a l _ S R _ P r o j e c t \ r e s u l t s \ e v a l \ r e s s h i f t \ t e s t \ r e c o n _ p e r _ i m a g e . c s v  
 R e s S h i f t   r e c o n   s u m m a r y   C S V :   D : \ C S E _ 2 1 2 0 1 0 1 6 _ B R A C \ C O N F E R E N C E \ F e t a l _ S R _ P r o j e c t \ r e s u l t s \ e v a l \ r e s s h i f t \ t e s t \ r e c o n _ s u m m a r y . c s v  
 { ' m e t h o d ' :   ' L Q _ m i x e d   b a s e l i n e ' ,   ' P S N R ' :   2 5 . 3 8 4 9 4 3 8 1 1 9 6 7 0 9 4 ,   ' S S I M ' :   0 . 6 7 2 2 5 6 1 1 0 1 2 5 5 6 3 9 ,   ' L P I P S ' :   0 . 3 5 0 9 6 8 1 0 4 1 0 4 2 0 1 ,   ' D I S T S ' :   0 . 2 4 1 3 3 3 6 0 1 4 7 4 7 6 1 9 6 ,   ' M A E ' :   0 . 0 3 3 7 4 5 9 7 7 4 8 7 4 1 5 0 8 ,   ' E d g e S c o r e ' :   0 . 9 7 7 9 2 9 2 7 5 8 3 0 5 8 6 8 ,   ' N I Q E ' :   9 . 8 0 8 5 6 5 2 6 0 5 6 9 2 5 5 ,   ' B R I S Q U E ' :   6 9 . 1 6 8 0 6 5 5 9 2 4 4 7 9 2 }  
 { ' m e t h o d ' :   ' S R - o n l y ' ,   ' P S N R ' :   3 0 . 9 8 5 6 4 8 9 8 0 8 7 6 6 7 ,   ' S S I M ' :   0 . 8 5 6 4 1 1 4 9 4 4 4 9 2 9 2 2 ,   ' L P I P S ' :   0 . 1 7 4 9 4 1 9 8 7 0 4 7 5 9 2 8 ,   ' D I S T S ' :   0 . 1 7 4 0 8 8 4 6 1 7 9 6 4 4 2 6 6 ,   ' M A E ' :   0 . 0 1 7 6 5 2 0 5 7 2 2 3 0 2 1 9 8 5 ,   ' E d g e S c o r e ' :   0 . 9 8 5 7 0 0 1 5 5 8 9 3 9 6 1 5 ,   ' N I Q E ' :   7 . 4 5 2 2 6 3 4 3 7 9 0 6 9 0 1 ,   ' B R I S Q U E ' :   5 4 . 5 1 4 7 7 3 7 6 3 0 2 0 8 3 4 }  
 { ' m e t h o d ' :   ' Q u a l i t y - a w a r e   S R ' ,   ' P S N R ' :   3 0 . 5 8 0 4 1 9 7 2 5 7 1 5 4 3 ,   ' S S I M ' :   0 . 8 7 4 3 3 6 6 8 0 0 4 3 6 3 0 9 ,   ' L P I P S ' :   0 . 2 0 4 2 1 6 4 1 3 7 4 6 2 7 7 5 ,   ' D I S T S ' :   0 . 1 8 1 1 8 6 2 6 2 7 6 6 5 2 0 2 ,   ' M A E ' :   0 . 0 1 7 3 9 4 0 6 5 8 0 4 1 5 8 6 4 6 ,   ' E d g e S c o r e ' :   0 . 9 8 4 9 7 6 4 5 0 9 9 9 5 7 7 9 ,   ' N I Q E ' :   7 . 6 1 7 8 4 0 3 4 0 9 3 2 2 1 1 ,   ' B R I S Q U E ' :   5 7 . 5 1 5 7 2 2 6 5 6 2 5 }  
 { ' m e t h o d ' :   ' S R - o n l y   +   L P I P S   +   G A N ' ,   ' P S N R ' :   2 7 . 8 6 3 6 9 6 4 9 6 4 0 5 8 5 8 ,   ' S S I M ' :   0 . 7 8 1 3 6 0 4 1 1 3 4 8 3 1 1 6 ,   ' L P I P S ' :   0 . 1 2 3 7 3 4 3 6 0 2 8 7 5 8 6 8 4 ,   ' D I S T S ' :   0 . 1 7 7 2 9 8 7 0 6 3 7 2 5 7 8 9 4 ,   ' M A E ' :   0 . 0 2 5 1 1 7 0 8 2 3 2 2 6 3 7 2 4 ,   ' E d g e S c o r e ' :   0 . 9 8 1 2 6 2 2 7 2 5 9 6 3 5 9 2 ,   ' N I Q E ' :   6 . 8 9 8 3 1 9 7 5 9 3 6 8 8 9 6 5 ,   ' B R I S Q U E ' :   2 9 . 0 8 6 9 1 6 5 0 3 9 0 6 2 5 }  
 { ' m e t h o d ' :   ' Q u a l i t y - a w a r e   S R   +   L P I P S   +   G A N ' ,   ' P S N R ' :   2 8 . 7 4 3 5 2 6 6 8 4 4 5 4 6 9 3 ,   ' S S I M ' :   0 . 8 2 6 6 1 7 2 2 9 8 5 1 0 0 8 3 ,   ' L P I P S ' :   0 . 1 1 3 0 7 3 2 4 2 4 5 5 7 2 0 9 ,   ' D I S T S ' :   0 . 1 6 7 1 0 8 1 0 1 8 4 4 7 8 7 6 ,   ' M A E ' :   0 . 0 2 1 9 5 8 1 0 2 2 3 6 1 9 1 4 3 ,   ' E d g e S c o r e ' :   0 . 9 8 2 6 6 7 1 4 4 1 3 9 6 0 7 8 ,   ' N I Q E ' :   6 . 1 6 0 1 6 9 4 2 6 6 0 0 1 3 8 ,   ' B R I S Q U E ' :   2 7 . 4 3 0 3 4 0 1 6 9 2 7 0 8 3 5 }  
 { ' m e t h o d ' :   ' S w i n I R - L i g h t   P h a s e   A ' ,   ' P S N R ' :   2 9 . 7 2 7 7 0 6 3 8 9 4 6 0 1 7 ,   ' S S I M ' :   0 . 8 4 6 2 3 0 4 3 9 2 5 7 9 0 7 1 ,   ' L P I P S ' :   0 . 1 0 0 9 1 4 5 6 2 1 2 1 0 3 3 6 7 ,   ' D I S T S ' :   0 . 1 5 0 8 6 8 2 2 0 7 2 6 6 4 8 9 6 ,   ' M A E ' :   0 . 0 1 9 4 2 8 1 2 7 8 2 4 0 1 8 4 ,   ' E d g e S c o r e ' :   0 . 9 8 4 1 5 9 0 3 8 4 6 4 2 2 8 4 ,   ' N I Q E ' :   6 . 0 7 2 5 0 0 8 5 5 1 2 7 9 7 ,   ' B R I S Q U E ' :   3 9 . 5 1 7 2 5 3 4 1 7 9 6 8 7 5 }  
 { ' m e t h o d ' :   ' R e s S h i f t ' ,   ' P S N R ' :   2 6 . 5 7 1 6 3 8 9 1 3 5 4 6 8 8 4 ,   ' S S I M ' :   0 . 7 2 6 0 5 0 6 1 9 2 5 6 1 4 7 3 ,   ' L P I P S ' :   0 . 2 2 2 7 2 9 4 3 1 3 9 0 7 6 2 3 2 ,   ' D I S T S ' :   0 . 2 0 9 3 5 8 7 5 4 5 5 5 3 8 4 3 3 ,   ' M A E ' :   0 . 0 2 9 0 7 1 9 7 6 7 0 3 9 0 2 0 8 5 ,   ' E d g e S c o r e ' :   0 . 9 8 0 5 6 2 5 5 6 5 8 4 6 7 6 1 ,   ' N I Q E ' :   6 . 4 8 5 4 8 0 3 0 0 5 8 5 4 2 8 5 ,   ' B R I S Q U E ' :   4 0 . 0 4 5 1 9 2 0 5 7 2 9 1 6 6 5 }  
 

## Step 7: scripts/46_downstream_utility_eval.py
p y t h o n . e x e   :   C o u l d   n o t   s a v e   f o n t _ m a n a g e r   c a c h e   [ E r r n o   1 3 ]   P e r m i s s i o n   d e n i e d :    
 ' C : \ \ U s e r s \ \ c a p t i \ \ . m a t p l o t l i b \ \ f o n t l i s t - v 3 . 1 1 . 0 . j s o n . m a t p l o t l i b - l o c k '  
 A t   l i n e : 5   c h a r : 1  
 +   &   $ p y   s c r i p t s \ 4 6 _ d o w n s t r e a m _ u t i l i t y _ e v a l . p y   2 > & 1   |   T e e - O b j e c t   - F i l e P a   . . .  
 +   ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  
         +   C a t e g o r y I n f o                     :   N o t S p e c i f i e d :   ( C o u l d   n o t   s a v e   . . . a t p l o t l i b - l o c k ' : S t r i n g )   [ ] ,   R e m o t e E x c e p t i o n  
         +   F u l l y Q u a l i f i e d E r r o r I d   :   N a t i v e C o m m a n d E r r o r  
    
 U s i n g   d e v i c e :   c u d a  
 L o a d e d   f r o z e n   H C 1 8   U - N e t :   e p o c h = 2 6   t h r e s h o l d = 0 . 5 0   i m a g e _ s i z e = 2 5 6  
 L o a d i n g   S R   c h e c k p o i n t s . . .  
 C o m p u t i n g   q u a l i t y - a w a r e   f e a t u r e   n o r m a l i z a t i o n   f r o m   t h e   t r a i n   s p l i t . . .  
 E v a l u a t i n g   1 5 0   H C 1 8   t e s t   i m a g e s   a c r o s s   9   i n p u t s . . .  
 S k i p p e d   m a s k s :   u n r e a d a b l e = 0 ,   e m p t y _ a f t e r _ p r e p r o c e s s i n g = 2  
  
 D o w n s t r e a m   u t i l i t y   s u m m a r y  
                                     m e t h o d     D i c e _ m e a n     D i c e _ s t d     I o U _ m e a n     A r e a E r r _ m e a n     D i a m E r r _ m e a n  
                                             H R       0 . 8 8 9 6 5 7     0 . 1 4 1 0 0 1     0 . 8 2 2 4 7 5             0 . 1 9 9 1 0 9           1 6 . 2 0 5 9 7 3  
                                 L Q _ m i x e d       0 . 8 8 4 9 9 0     0 . 1 4 1 4 3 2     0 . 8 1 5 0 2 8             0 . 2 0 1 9 0 3           1 6 . 3 7 7 5 3 0  
                                   S R - o n l y       0 . 8 8 5 3 2 7     0 . 1 4 2 3 9 3     0 . 8 1 5 8 1 8             0 . 2 0 9 6 6 4           1 6 . 5 7 6 9 0 6  
             S R - o n l y + p e r c e p t u a l       0 . 8 8 7 2 8 6     0 . 1 4 0 7 2 5     0 . 8 1 8 4 4 2             0 . 2 0 2 4 5 9           1 6 . 3 4 1 4 0 3  
                       Q u a l i t y - a w a r e       0 . 8 8 4 6 8 8     0 . 1 4 2 3 7 1     0 . 8 1 4 7 7 9             0 . 2 1 1 4 1 0           1 6 . 6 5 2 6 5 3  
 Q u a l i t y - a w a r e + p e r c e p t u a l       0 . 8 8 5 8 1 7     0 . 1 4 1 2 8 0     0 . 8 1 6 2 3 6             0 . 2 0 4 2 5 8           1 6 . 4 8 9 1 2 6  
         S w i n I R - L i g h t   P h a s e   A       0 . 8 8 7 7 0 2     0 . 1 4 1 1 0 4     0 . 8 1 9 2 7 1             0 . 2 0 2 8 1 5           1 6 . 3 3 9 0 6 7  
                                 R e s S h i f t       0 . 8 8 4 9 3 8     0 . 1 4 1 8 5 0     0 . 8 1 5 0 0 3             0 . 2 0 8 0 2 7           1 6 . 6 3 1 6 4 0  
     C N N - R e s i d u a l   D i f f u s i o n       0 . 8 8 5 4 3 0     0 . 1 4 1 2 9 8     0 . 8 1 5 6 4 4             0 . 2 0 7 8 6 3           1 6 . 5 5 0 2 3 2  
  
 W r o t e   p e r - i m a g e   m e t r i c s :   r e s u l t s \ d o w n s t r e a m _ u t i l i t y _ p e r _ i m a g e . c s v  
 W r o t e   s u m m a r y   m e t r i c s :   r e s u l t s \ d o w n s t r e a m _ u t i l i t y _ s u m m a r y . c s v  
 W r o t e   p r e d i c t e d   m a s k s   u n d e r :   r e s u l t s \ d o w n s t r e a m _ u t i l i t y _ m a s k s  
 W r o t e   S w i n I R   d o w n s t r e a m   r o w s :   r e s u l t s \ e v a l \ s w i n i r _ l i g h t _ p h a s e _ a \ t e s t \ d o w n s t r e a m _ p e r _ i m a g e . c s v  
 W r o t e   R e s S h i f t   d o w n s t r e a m   r o w s :   r e s u l t s \ e v a l \ r e s s h i f t \ t e s t \ d o w n s t r e a m _ p e r _ i m a g e . c s v  
  
 A C C E P T A N C E :   F A I L  
 L Q _ m i x e d   D i c e _ m e a n :   0 . 8 8 4 9 9 0  
 S R - o n l y + p e r c e p t u a l   D i c e _ m e a n :   0 . 8 8 7 2 8 6   ( d e l t a   v s   L Q :   + 0 . 0 0 2 2 9 6 )  
 Q u a l i t y - a w a r e + p e r c e p t u a l   D i c e _ m e a n :   0 . 8 8 5 8 1 7   ( d e l t a   v s   L Q :   + 0 . 0 0 0 8 2 7 ;   d e l t a   v s   S R - o n l y + p e r c e p t u a l :   - 0 . 0 0 1 4 6 9 )  
 

## Step 7: scripts/47_hard_subset_analysis.py
p y t h o n . e x e   :   C o u l d   n o t   s a v e   f o n t _ m a n a g e r   c a c h e   [ E r r n o   1 3 ]   P e r m i s s i o n   d e n i e d :    
 ' C : \ \ U s e r s \ \ c a p t i \ \ . m a t p l o t l i b \ \ f o n t l i s t - v 3 . 1 1 . 0 . j s o n . m a t p l o t l i b - l o c k '  
 A t   l i n e : 5   c h a r : 1  
 +   &   $ p y   s c r i p t s \ 4 7 _ h a r d _ s u b s e t _ a n a l y s i s . p y   2 > & 1   |   T e e - O b j e c t   - F i l e P a t h     . . .  
 +   ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  
         +   C a t e g o r y I n f o                     :   N o t S p e c i f i e d :   ( C o u l d   n o t   s a v e   . . . a t p l o t l i b - l o c k ' : S t r i n g )   [ ] ,   R e m o t e E x c e p t i o n  
         +   F u l l y Q u a l i f i e d E r r o r I d   :   N a t i v e C o m m a n d E r r o r  
    
 S T E P   3 a   -   H A R D - S U B S E T   D O W N S T R E A M   A N A L Y S I S  
 M e m b e r s h i p   C S V :   r e s u l t s \ d o w n s t r e a m _ h a r d _ s u b s e t _ m e m b e r s h i p . c s v  
 S u m m a r y   C S V :   r e s u l t s \ d o w n s t r e a m _ h a r d _ s u b s e t _ s u m m a r y . c s v  
 F i g u r e :   r e s u l t s \ d o w n s t r e a m _ h a r d _ s u b s e t _ f i g u r e s \ d i c e _ h a r d _ v s _ f u l l . p n g  
 S w i n I R   h a r d   s u b s e t s   C S V :   r e s u l t s \ e v a l \ s w i n i r _ l i g h t _ p h a s e _ a \ t e s t \ h a r d _ s u b s e t s . c s v  
 R e s S h i f t   h a r d   s u b s e t s   C S V :   r e s u l t s \ e v a l \ r e s s h i f t \ t e s t \ h a r d _ s u b s e t s . c s v  
 S u b s e t   s i z e s :   { ' f u l l _ t e s t ' :   1 4 8 ,   ' b o t t o m 2 5 _ l q _ d i c e ' :   3 7 ,   ' l q _ d i c e _ l t _ 0 _ 8 5 ' :   4 0 ,   ' t o p 2 5 _ l q _ l p i p s ' :   3 7 }  
  
 H a r d - s u b s e t   s u m m a r y  
                     s u b s e t                                       m e t h o d       N     D i c e _ m e a n     I o U _ m e a n     A r e a E r r _ m e a n     D i a m E r r _ m e a n     D i c e _ d e l t a _ v s _ L Q     I o U _ d e l t a _ v s _ L Q     A r e a E r r _ d e l t a _ v s _ L Q     D i a m E r r _ d e l t a _ v s _ L Q  
               f u l l _ t e s t                                   L Q _ m i x e d   1 4 8       0 . 8 8 4 9 9 0     0 . 8 1 5 0 2 8             0 . 2 0 1 9 0 3           1 6 . 3 7 7 5 3 0                     0 . 0 0 0 0 0 0                   0 . 0 0 0 0 0 0                           0 . 0 0 0 0 0 0                           0 . 0 0 0 0 0 0  
               f u l l _ t e s t                                     S R - o n l y   1 4 8       0 . 8 8 5 3 2 7     0 . 8 1 5 8 1 8             0 . 2 0 9 6 6 4           1 6 . 5 7 6 9 0 6                     0 . 0 0 0 3 3 6                   0 . 0 0 0 7 8 9                           0 . 0 0 7 7 6 1                           0 . 1 9 9 3 7 6  
               f u l l _ t e s t                         Q u a l i t y - a w a r e   1 4 8       0 . 8 8 4 6 8 8     0 . 8 1 4 7 7 9             0 . 2 1 1 4 1 0           1 6 . 6 5 2 6 5 3                   - 0 . 0 0 0 3 0 2                 - 0 . 0 0 0 2 4 9                           0 . 0 0 9 5 0 7                           0 . 2 7 5 1 2 3  
               f u l l _ t e s t               S R - o n l y + p e r c e p t u a l   1 4 8       0 . 8 8 7 2 8 6     0 . 8 1 8 4 4 2             0 . 2 0 2 4 5 9           1 6 . 3 4 1 4 0 3                     0 . 0 0 2 2 9 6                   0 . 0 0 3 4 1 4                           0 . 0 0 0 5 5 6                         - 0 . 0 3 6 1 2 7  
               f u l l _ t e s t   Q u a l i t y - a w a r e + p e r c e p t u a l   1 4 8       0 . 8 8 5 8 1 7     0 . 8 1 6 2 3 6             0 . 2 0 4 2 5 8           1 6 . 4 8 9 1 2 6                     0 . 0 0 0 8 2 7                   0 . 0 0 1 2 0 8                           0 . 0 0 2 3 5 5                           0 . 1 1 1 5 9 5  
               f u l l _ t e s t       C N N - R e s i d u a l   D i f f u s i o n   1 4 8       0 . 8 8 5 4 3 0     0 . 8 1 5 6 4 4             0 . 2 0 7 8 6 3           1 6 . 5 5 0 2 3 2                     0 . 0 0 0 4 4 0                   0 . 0 0 0 6 1 6                           0 . 0 0 5 9 6 1                           0 . 1 7 2 7 0 2  
               f u l l _ t e s t           S w i n I R - L i g h t   P h a s e   A   1 4 8       0 . 8 8 7 7 0 2     0 . 8 1 9 2 7 1             0 . 2 0 2 8 1 5           1 6 . 3 3 9 0 6 7                     0 . 0 0 2 7 1 2                   0 . 0 0 4 2 4 3                           0 . 0 0 0 9 1 2                         - 0 . 0 3 8 4 6 3  
               f u l l _ t e s t                                   R e s S h i f t   1 4 8       0 . 8 8 4 9 3 8     0 . 8 1 5 0 0 3             0 . 2 0 8 0 2 7           1 6 . 6 3 1 6 4 0                   - 0 . 0 0 0 0 5 3                 - 0 . 0 0 0 0 2 5                           0 . 0 0 6 1 2 5                           0 . 2 5 4 1 0 9  
 b o t t o m 2 5 _ l q _ d i c e                                   L Q _ m i x e d     3 7       0 . 7 0 4 8 0 0     0 . 5 6 5 5 1 0             0 . 5 5 8 6 8 4           4 5 . 7 2 9 0 2 1                     0 . 0 0 0 0 0 0                   0 . 0 0 0 0 0 0                           0 . 0 0 0 0 0 0                           0 . 0 0 0 0 0 0  
 b o t t o m 2 5 _ l q _ d i c e                                     S R - o n l y     3 7       0 . 7 0 3 9 8 8     0 . 5 6 5 3 5 4             0 . 5 9 0 7 4 7           4 6 . 7 4 3 3 9 4                   - 0 . 0 0 0 8 1 2                 - 0 . 0 0 0 1 5 6                           0 . 0 3 2 0 6 3                           1 . 0 1 4 3 7 3  
 b o t t o m 2 5 _ l q _ d i c e                         Q u a l i t y - a w a r e     3 7       0 . 7 0 5 3 3 0     0 . 5 6 7 3 1 1             0 . 5 8 8 7 2 1           4 6 . 5 9 2 3 9 1                     0 . 0 0 0 5 3 1                   0 . 0 0 1 8 0 1                           0 . 0 3 0 0 3 7                           0 . 8 6 3 3 7 0  
 b o t t o m 2 5 _ l q _ d i c e               S R - o n l y + p e r c e p t u a l     3 7       0 . 7 0 8 8 5 2     0 . 5 7 1 2 9 5             0 . 5 6 7 1 8 4           4 6 . 0 1 2 3 9 1                     0 . 0 0 4 0 5 2                   0 . 0 0 5 7 8 5                           0 . 0 0 8 5 0 0                           0 . 2 8 3 3 7 1  
 b o t t o m 2 5 _ l q _ d i c e   Q u a l i t y - a w a r e + p e r c e p t u a l     3 7       0 . 7 0 6 1 1 4     0 . 5 6 7 8 2 6             0 . 5 7 2 5 5 7           4 6 . 5 8 2 5 9 4                     0 . 0 0 1 3 1 5                   0 . 0 0 2 3 1 6                           0 . 0 1 3 8 7 3                           0 . 8 5 3 5 7 3  
 b o t t o m 2 5 _ l q _ d i c e       C N N - R e s i d u a l   D i f f u s i o n     3 7       0 . 7 0 8 0 1 5     0 . 5 7 0 3 6 6             0 . 5 7 5 1 3 4           4 6 . 1 9 0 2 3 9                     0 . 0 0 3 2 1 5                   0 . 0 0 4 8 5 5                           0 . 0 1 6 4 5 0                           0 . 4 6 1 2 1 9  
 b o t t o m 2 5 _ l q _ d i c e           S w i n I R - L i g h t   P h a s e   A     3 7       0 . 7 0 8 4 6 0     0 . 5 7 0 6 3 3             0 . 5 7 1 1 9 5           4 6 . 2 2 6 4 6 7                     0 . 0 0 3 6 6 1                   0 . 0 0 5 1 2 3                           0 . 0 1 2 5 1 1                           0 . 4 9 7 4 4 6  
 b o t t o m 2 5 _ l q _ d i c e                                   R e s S h i f t     3 7       0 . 7 0 2 7 7 1     0 . 5 6 3 1 7 2             0 . 5 8 2 6 0 7           4 6 . 7 8 6 1 7 3                   - 0 . 0 0 2 0 2 9                 - 0 . 0 0 2 3 3 8                           0 . 0 2 3 9 2 3                           1 . 0 5 7 1 5 3  
   l q _ d i c e _ l t _ 0 _ 8 5                                   L Q _ m i x e d     4 0       0 . 7 1 5 2 9 4     0 . 5 7 7 9 3 9             0 . 5 3 4 2 9 0           4 4 . 0 9 1 2 5 1                     0 . 0 0 0 0 0 0                   0 . 0 0 0 0 0 0                           0 . 0 0 0 0 0 0                           0 . 0 0 0 0 0 0  
   l q _ d i c e _ l t _ 0 _ 8 5                                     S R - o n l y     4 0       0 . 7 1 5 2 0 0     0 . 5 7 8 8 1 8             0 . 5 6 1 4 0 4           4 4 . 8 9 9 1 6 4                   - 0 . 0 0 0 0 9 4                   0 . 0 0 0 8 7 9                           0 . 0 2 7 1 1 3                           0 . 8 0 7 9 1 3  
   l q _ d i c e _ l t _ 0 _ 8 5                         Q u a l i t y - a w a r e     4 0       0 . 7 1 5 4 3 6     0 . 5 7 9 0 8 6             0 . 5 6 2 2 1 1           4 4 . 9 1 1 7 0 9                     0 . 0 0 0 1 4 3                   0 . 0 0 1 1 4 7                           0 . 0 2 7 9 2 0                           0 . 8 2 0 4 5 8  
   l q _ d i c e _ l t _ 0 _ 8 5               S R - o n l y + p e r c e p t u a l     4 0       0 . 7 2 1 0 9 3     0 . 5 8 6 6 5 0             0 . 5 3 9 6 6 3           4 4 . 2 3 1 6 2 6                     0 . 0 0 5 7 9 9                   0 . 0 0 8 7 1 1                           0 . 0 0 5 3 7 2                           0 . 1 4 0 3 7 5  
   l q _ d i c e _ l t _ 0 _ 8 5   Q u a l i t y - a w a r e + p e r c e p t u a l     4 0       0 . 7 1 8 2 0 1     0 . 5 8 2 8 1 1             0 . 5 4 4 1 8 0           4 4 . 7 3 1 2 6 6                     0 . 0 0 2 9 0 7                   0 . 0 0 4 8 7 2                           0 . 0 0 9 8 9 0                           0 . 6 4 0 0 1 5  
   l q _ d i c e _ l t _ 0 _ 8 5       C N N - R e s i d u a l   D i f f u s i o n     4 0       0 . 7 1 7 2 7 0     0 . 5 8 0 9 7 0             0 . 5 5 1 2 8 2           4 4 . 6 3 2 2 3 7                     0 . 0 0 1 9 7 7                   0 . 0 0 3 0 3 1                           0 . 0 1 6 9 9 1                           0 . 5 4 0 9 8 6  
   l q _ d i c e _ l t _ 0 _ 8 5           S w i n I R - L i g h t   P h a s e   A     4 0       0 . 7 2 0 7 9 0     0 . 5 8 6 1 2 9             0 . 5 4 3 1 2 8           4 4 . 4 1 0 2 8 8                     0 . 0 0 5 4 9 6                   0 . 0 0 8 1 9 0                           0 . 0 0 8 8 3 8                           0 . 3 1 9 0 3 7  
   l q _ d i c e _ l t _ 0 _ 8 5                                   R e s S h i f t     4 0       0 . 7 1 5 2 7 4     0 . 5 7 8 8 4 6             0 . 5 5 3 7 0 6           4 4 . 9 5 1 1 0 1                   - 0 . 0 0 0 0 2 0                   0 . 0 0 0 9 0 7                           0 . 0 1 9 4 1 6                           0 . 8 5 9 8 5 1  
     t o p 2 5 _ l q _ l p i p s                                   L Q _ m i x e d     3 7       0 . 8 6 3 4 6 9     0 . 7 7 5 9 4 0             0 . 2 3 5 1 4 1           1 7 . 7 4 4 5 7 7                     0 . 0 0 0 0 0 0                   0 . 0 0 0 0 0 0                           0 . 0 0 0 0 0 0                           0 . 0 0 0 0 0 0  
     t o p 2 5 _ l q _ l p i p s                                     S R - o n l y     3 7       0 . 8 5 9 7 2 9     0 . 7 7 1 1 2 1             0 . 2 7 2 4 2 3           1 8 . 8 9 8 5 0 2                   - 0 . 0 0 3 7 4 0                 - 0 . 0 0 4 8 2 0                           0 . 0 3 7 2 8 2                           1 . 1 5 3 9 2 5  
     t o p 2 5 _ l q _ l p i p s                         Q u a l i t y - a w a r e     3 7       0 . 8 5 7 6 7 2     0 . 7 6 7 7 6 1             0 . 2 7 6 7 4 7           1 9 . 1 6 4 3 6 4                   - 0 . 0 0 5 7 9 6                 - 0 . 0 0 8 1 7 9                           0 . 0 4 1 6 0 6                           1 . 4 1 9 7 8 7  
     t o p 2 5 _ l q _ l p i p s               S R - o n l y + p e r c e p t u a l     3 7       0 . 8 6 7 7 8 0     0 . 7 8 2 1 7 0             0 . 2 4 4 1 2 3           1 7 . 8 7 1 1 5 0                     0 . 0 0 4 3 1 1                   0 . 0 0 6 2 3 0                           0 . 0 0 8 9 8 2                           0 . 1 2 6 5 7 3  
     t o p 2 5 _ l q _ l p i p s   Q u a l i t y - a w a r e + p e r c e p t u a l     3 7       0 . 8 6 5 9 0 2     0 . 7 7 9 4 8 7             0 . 2 4 7 3 3 6           1 8 . 1 1 1 1 1 9                     0 . 0 0 2 4 3 4                   0 . 0 0 3 5 4 7                           0 . 0 1 2 1 9 5                           0 . 3 6 6 5 4 2  
     t o p 2 5 _ l q _ l p i p s       C N N - R e s i d u a l   D i f f u s i o n     3 7       0 . 8 6 2 2 0 5     0 . 7 7 3 6 6 7             0 . 2 6 1 8 1 1           1 8 . 6 5 9 1 4 1                   - 0 . 0 0 1 2 6 4                 - 0 . 0 0 2 2 7 4                           0 . 0 2 6 6 7 0                           0 . 9 1 4 5 6 4  
     t o p 2 5 _ l q _ l p i p s           S w i n I R - L i g h t   P h a s e   A     3 7       0 . 8 6 7 8 8 2     0 . 7 8 2 1 7 7             0 . 2 4 5 6 8 4           1 7 . 8 3 5 1 3 2                     0 . 0 0 4 4 1 3                   0 . 0 0 6 2 3 6                           0 . 0 1 0 5 4 3                           0 . 0 9 0 5 5 5  
     t o p 2 5 _ l q _ l p i p s                                   R e s S h i f t     3 7       0 . 8 6 1 5 2 6     0 . 7 7 3 0 1 6             0 . 2 6 4 3 2 1           1 8 . 7 6 4 2 1 6                   - 0 . 0 0 1 9 4 3                 - 0 . 0 0 2 9 2 5                           0 . 0 2 9 1 8 0                           1 . 0 1 9 6 3 9  
  
 S u b s e t   m e m b e r s h i p   t h r e s h o l d s  
   d i c e _ q 2 5 _ t h r e s h o l d     l p i p s _ q 7 5 _ t h r e s h o l d  
                       0 . 8 3 5 3 9 7                           0 . 3 7 8 0 3 8  
 

## Step 7: scripts/48_significance_report.py
B o o t s t r a p   m e t r i c   s u m m a r y :   r e s u l t s \ b o o t s t r a p _ m e t r i c _ s u m m a r y . c s v  
 S i g n i f i c a n c e   r e p o r t :   r e s u l t s \ s i g n i f i c a n c e _ r e p o r t . c s v  
 S T E P   3 b   -   S I G N I F I C A N C E   S U M M A R Y  
 T o t a l   t e s t s :   3 9  
 B o n f e r r o n i - s i g n i f i c a n t   t e s t s :   1 7  
 |   c o m p a r i s o n   |   m e t r i c   |   s u b s e t   |   d e l t a   ( B - A )   |   p _ b o n f   |  
 | - - - | - - - : | - - - | - - - : | - - - : |  
 |   S R - o n l y   - >   S R - o n l y + p e r c e p t u a l   |   L P I P S   |   f u l l _ t e s t   |   - 0 . 0 5 0 9 6 2   |   3 . 5 4 8 e - 2 0   |  
 |   S R - o n l y + p e r c e p t u a l   - >   Q u a l i t y - a w a r e + p e r c e p t u a l   |   L P I P S   |   f u l l _ t e s t   |   - 0 . 0 1 0 8 3 9   |   1 . 7 5 7 e - 1 4   |  
 |   S R - o n l y + p e r c e p t u a l   - >   S w i n I R - L i g h t   P h a s e   A   |   L P I P S   |   f u l l _ t e s t   |   - 0 . 0 2 3 0 2 5   |   3 . 6 3 3 e - 1 9   |  
 |   Q u a l i t y - a w a r e + p e r c e p t u a l   - >   S w i n I R - L i g h t   P h a s e   A   |   L P I P S   |   f u l l _ t e s t   |   - 0 . 0 1 2 1 8 6   |   3 . 9 8 7 e - 1 0   |  
 |   Q u a l i t y - a w a r e + p e r c e p t u a l   - >   S w i n I R - L i g h t   P h a s e   A   |   D i c e   |   f u l l _ t e s t   |   0 . 0 0 1 8 8 5   |   2 . 4 5 2 e - 0 2   |  
 |   S R - o n l y   - >   S R - o n l y + p e r c e p t u a l   |   L P I P S   |   b o t t o m 2 5 _ l q _ d i c e   |   - 0 . 0 4 9 9 2 0   |   2 . 8 2 5 e - 0 6   |  
 |   S R - o n l y + p e r c e p t u a l   - >   Q u a l i t y - a w a r e + p e r c e p t u a l   |   L P I P S   |   b o t t o m 2 5 _ l q _ d i c e   |   - 0 . 0 1 2 1 3 3   |   8 . 6 4 7 e - 0 4   |  
 |   S R - o n l y + p e r c e p t u a l   - >   S w i n I R - L i g h t   P h a s e   A   |   L P I P S   |   b o t t o m 2 5 _ l q _ d i c e   |   - 0 . 0 2 3 0 5 3   |   5 . 1 3 0 e - 0 7   |  
 |   Q u a l i t y - a w a r e + p e r c e p t u a l   - >   S w i n I R - L i g h t   P h a s e   A   |   L P I P S   |   b o t t o m 2 5 _ l q _ d i c e   |   - 0 . 0 1 0 9 2 0   |   1 . 0 2 8 e - 0 3   |  
 |   S R - o n l y   - >   S R - o n l y + p e r c e p t u a l   |   L P I P S   |   l q _ d i c e _ l t _ 0 _ 8 5   |   - 0 . 0 5 1 1 5 9   |   4 . 0 7 1 e - 0 7   |  
 |   S R - o n l y + p e r c e p t u a l   - >   Q u a l i t y - a w a r e + p e r c e p t u a l   |   L P I P S   |   l q _ d i c e _ l t _ 0 _ 8 5   |   - 0 . 0 1 1 7 8 2   |   2 . 2 8 0 e - 0 4   |  
 |   S R - o n l y + p e r c e p t u a l   - >   S w i n I R - L i g h t   P h a s e   A   |   L P I P S   |   l q _ d i c e _ l t _ 0 _ 8 5   |   - 0 . 0 2 2 7 5 5   |   7 . 5 8 4 e - 0 8   |  
 |   Q u a l i t y - a w a r e + p e r c e p t u a l   - >   S w i n I R - L i g h t   P h a s e   A   |   L P I P S   |   l q _ d i c e _ l t _ 0 _ 8 5   |   - 0 . 0 1 0 9 7 3   |   4 . 7 6 2 e - 0 4   |  
 |   S R - o n l y   - >   S R - o n l y + p e r c e p t u a l   |   L P I P S   |   t o p 2 5 _ l q _ l p i p s   |   - 0 . 0 7 5 1 8 3   |   5 . 6 7 5 e - 1 0   |  
 |   S R - o n l y + p e r c e p t u a l   - >   Q u a l i t y - a w a r e + p e r c e p t u a l   |   L P I P S   |   t o p 2 5 _ l q _ l p i p s   |   - 0 . 0 0 7 5 3 9   |   1 . 7 0 2 e - 0 3   |  
 |   S R - o n l y + p e r c e p t u a l   - >   S w i n I R - L i g h t   P h a s e   A   |   L P I P S   |   t o p 2 5 _ l q _ l p i p s   |   - 0 . 0 1 6 9 3 4   |   7 . 7 8 0 e - 0 5   |  
 |   L Q _ m i x e d   - >   Q u a l i t y - a w a r e + p e r c e p t u a l   |   L P I P S   |   f u l l _ t e s t   |   - 0 . 2 3 7 6 3 8   |   1 . 9 1 1 e - 2 4   |  
 

## Step 7 post-check: add requested ResShift significance rows without editing script 48
U p d a t e d   r e s u l t s \ s i g n i f i c a n c e _ r e p o r t . c s v   w i t h   4   R e s S h i f t   r o w s ;   B o n f e r r o n i   d e n o m i n a t o r = 4 3  
                         m e t h o d _ A   m e t h o d _ B   m e t r i c         s u b s e t           d e l t a                 p _ r a w     n _ p a i r s  
 S w i n I R - L i g h t   P h a s e   A   R e s S h i f t     L P I P S   f u l l _ t e s t     0 . 1 2 1 8 1 5   2 . 2 9 9 6 0 6 e - 2 6             1 5 0  
 S w i n I R - L i g h t   P h a s e   A   R e s S h i f t       D i c e   f u l l _ t e s t   - 0 . 0 0 2 7 6 4   1 . 0 0 7 3 5 3 e - 0 4             1 4 8  
                         L Q _ m i x e d   R e s S h i f t     L P I P S   f u l l _ t e s t   - 0 . 1 2 8 2 3 9   2 . 4 9 2 1 3 2 e - 2 6             1 5 0  
                         L Q _ m i x e d   R e s S h i f t       D i c e   f u l l _ t e s t   - 0 . 0 0 0 0 5 3   2 . 6 5 3 4 2 6 e - 0 1             1 4 8  
 