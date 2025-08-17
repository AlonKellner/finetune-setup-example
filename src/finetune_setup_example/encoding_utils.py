"""Encoding utils."""

import base64


def booleans_to_s3_suffix(bool_list: list[bool]) -> str:
    """
    Convert a list of booleans into a unique, S3-compatible suffix string.

    The output string will only contain lowercase letters and numbers, making it
    safe for use in S3 bucket names.

    Args:
        bool_list: A list of boolean values (True or False).

    Returns
    -------
        An S3-compatible string suffix representing the list of booleans.

    Raises
    ------
        TypeError: If the input is not a list of booleans.

    The function works by:
    1. Packing the booleans into bytes (8 booleans per byte).
    2. Encoding these bytes using Base32, which uses an alphabet of A-Z and 2-7.
    3. Converting the resulting string to lowercase.
    4. Removing the padding character ('=') from the end.
    """
    # --- Input Validation ---
    if not isinstance(bool_list, list) or not all(
        isinstance(b, bool) for b in bool_list
    ):
        raise TypeError("Input must be a list of booleans.")

    if not bool_list:
        return ""

    # --- Packing Booleans into Bytes ---
    byte_array = bytearray()

    # Iterate through the boolean list in chunks of 8
    for i in range(0, len(bool_list), 8):
        byte = 0
        # Get a chunk of up to 8 booleans from the list
        chunk = bool_list[i : i + 8]

        # Pack the booleans from the chunk into a single byte
        for index, bit in enumerate(chunk):
            if bit:
                byte |= 1 << (7 - index)

        byte_array.append(byte)

    packed_bytes = bytes(byte_array)

    # --- Base32 Encoding for S3 Compatibility ---
    # 1. Encode the bytes using Base32
    base32_bytes = base64.b32encode(packed_bytes)

    # 2. Decode to a standard string
    base32_string = base32_bytes.decode("utf-8")

    # 3. Convert to lowercase to meet S3 bucket naming rules
    s3_safe_string = base32_string.lower()

    # 4. Remove padding characters ('=') which are not allowed in S3 names
    s3_suffix = s3_safe_string.rstrip("=")

    return f"{len(bool_list)}-{s3_suffix}"


if __name__ == "__main__":
    [
        print(booleans_to_s3_suffix([a, b, c] * 100))
        for a in [False, True]
        for b in [False, True]
        for c in [False, True]
    ]
