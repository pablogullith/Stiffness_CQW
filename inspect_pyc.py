import marshal, dis

with open('__pycache__/Figura02.cpython-314.pyc', 'rb') as f:
    f.seek(16) # Skip header for Python 3.14 (or 16 bytes)
    code = marshal.load(f)

print("Code co_consts:")
for c in code.co_consts:
    if isinstance(c, (str, tuple, list, dict)):
        print("  CONST:", repr(c)[:120])

print("\nCode co_names:")
print(code.co_names)

