def text_to_hex_converterGM(input_text: str) -> str:

    # Requesta string if the input is empty
    if not input_text:
        return "Please input a string."

    # Encode the input text into UTF-8 to handle extended characters
    byte_data = input_text.encode('utf-8')

    # Convert the bytes to hexadecimal and return it
    hex_output = byte_data.hex()

    return hex_output
