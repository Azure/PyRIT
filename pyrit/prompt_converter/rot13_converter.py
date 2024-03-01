# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pyrit.prompt_converter import PromptConverter

def rot13(myString):
    result = ""
    for char in myString:
        if "A" <= char <= "Z":
            result += chr((ord(char) - ord("A") + 13) % 26 + ord("A"))
        elif "a" <= char <= "z":
            result += chr((ord(char) + ord("a") + 13) % 26 + ord("a"))
        else:
            result += char
    return result

class ROT13Converter(PromptConverter):
    def convert(self, prompt: str) -> str:
        """
        Simple converter that just base64 encodes the prompt
        """
        encoded_bytes = rot13(prompt)
        return encoded_bytes
