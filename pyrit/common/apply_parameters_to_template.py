# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from typing import List


def apply_parameters_to_template(
    template: str,
    parameters: List[str],
    **kwargs
) -> str:
    """Inserts parameters into template.

    Args:
        template: the template, e.g., a prompt template with placeholders surrounded by double curly braces
        parameters: the list of parameters exist in the template
        **kwargs: the key-value pairs for parameters

    Returns:
        A new prompt following the template
    
    Raises:
        ValueError: If the parameters are invalid or missing in the template
    """
    final_prompt = template
    for key, value in kwargs.items():
        if key not in parameters:
            raise ValueError(f'Invalid parameters passed. [expected="{parameters}", actual="{kwargs}"]')
        # Matches field names within brackets {{ }}
        #  {{   key    }}
        #  ^^^^^^^^^^^^^^
        regex = f"{{{{ *{key} *}}}}"
        matches = re.findall(pattern=regex, string=final_prompt)
        if not matches:
            raise ValueError(
                f"No parameters matched, they might be missing in the template. "
                f'[expected="{parameters}", actual="{kwargs}"]'
            )
        final_prompt = re.sub(pattern=regex, string=final_prompt, repl=str(value))
    return final_prompt