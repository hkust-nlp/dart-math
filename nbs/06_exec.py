#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Code Execution
# 
# > Execute code in text efficiently and safely.
# 

# | hide
from nbdev.showdoc import *
from fastcore.test import *


from dart_math.exec import *


# ## Execution Code Cells as in Notebooks
# 

from dart_math.exec import *

exec_cells(
    [
        "print('Hey, Jude')",
        "print('Don\\'t make it bad')",
        "print('Take a sad song and make it better')",
    ]
)  # Only return the stdout and stderr of the last cell


show_doc(exec_cells, title_level=3)


# ## Unified Language & Code Context Configuration
# 

code_exec_cfg = CodeExecCfg.load_from_id_or_path("python")
code_exec_cfg.__dict__


code_exec_cfg


EG_LANG_CODE_CONTEXT = """
```python
print('Hey, Jude')
```

```output
Hey, Jude
```

Don't make it bad

```python
print('Take a sad song and make it better')
```

"""


code_exec_cfg.no_cells_todo(EG_LANG_CODE_CONTEXT)


code_exec_cfg.no_cells_todo(
    EG_LANG_CODE_CONTEXT + "```output\nTake a sad song and make it better```"
)


code_exec_cfg.extract_cells(EG_LANG_CODE_CONTEXT)


code_exec_cfg.wrap_output("Take a sad song and make it better")
# Usually appended with some newlines


show_doc(CodeExecCfg, title_level=3)


show_doc(CodeExecCfg.load_from_id_or_path, title_level=4)


show_doc(CodeExecCfg.no_cells_todo, title_level=4)


test_eq(
    code_exec_cfg.no_cells_todo(EG_LANG_CODE_CONTEXT), False
)  # 2 code cells but only 1 executed to output

test_eq(
    code_exec_cfg.no_cells_todo(
        EG_LANG_CODE_CONTEXT + "```output\nTake a sad song and make it better```"
    ),
    True,
)  # All the code cells have been executed


show_doc(CodeExecCfg.extract_cells, title_level=4)


test_eq(
    code_exec_cfg.extract_cells(EG_LANG_CODE_CONTEXT),
    [
        "print('Hey, Jude')",
        # "print('Don\\'t make it bad')",
        "print('Take a sad song and make it better')",
    ],
)


show_doc(CodeExecCfg.wrap_output, title_level=4)


test_eq(
    code_exec_cfg.wrap_output("Take a sad song and make it better"),
    "```output\nTake a sad song and make it better\n```",
)

