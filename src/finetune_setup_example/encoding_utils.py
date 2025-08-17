"""Encoding utils."""

import base64


def booleans_to_base64(bool_list: list[bool]) -> str:
    """
    Convert a list of booleans into a URL-safe base64 encoded string.

    Args:
        bool_list: A list of boolean values (True or False).

    Returns
    -------
        A base64 encoded string representing the list of booleans.

    Raises
    ------
        TypeError: If the input is not a list of booleans.

    The function works by packing 8 booleans at a time into a single byte.
    If the number of booleans is not a multiple of 8, the last byte is
    effectively padded with False values (zeros) to fill it out.
    """
    # --- Input Validation ---
    if not isinstance(bool_list, list) or not all(
        isinstance(b, bool) for b in bool_list
    ):
        raise TypeError("Input must be a list of booleans.")

    # --- Packing Booleans into Bytes ---
    byte_array = bytearray()

    # Iterate through the boolean list in chunks of 8
    for i in range(0, len(bool_list), 8):
        byte = 0
        # Get a chunk of up to 8 booleans from the list
        chunk = bool_list[i : i + 8]

        # Pack the booleans from the chunk into a single byte
        # The first boolean in the chunk corresponds to the most significant bit (MSB)
        for index, bit in enumerate(chunk):
            if bit:
                # Use a bitwise OR operation to set the appropriate bit in the byte.
                # (7 - index) ensures we go from MSB (left) to LSB (right).
                # For index 0, we shift 1 by 7 bits (10000000)
                # For index 7, we shift 1 by 0 bits (00000001)
                byte |= 1 << (7 - index)

        # Add the completed byte to our bytearray
        byte_array.append(byte)

    # Convert the bytearray into an immutable bytes object
    packed_bytes = bytes(byte_array)

    # --- Base64 Encoding ---
    # Encode the bytes object into a base64 bytes object
    base64_bytes = base64.b64encode(packed_bytes)

    # Decode the base64 bytes into a standard UTF-8 string for the final result
    base64_string = base64_bytes.decode("utf-8")

    return base64_string
