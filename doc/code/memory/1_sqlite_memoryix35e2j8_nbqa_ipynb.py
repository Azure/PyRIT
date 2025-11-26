# %%NBQA-CELL-SEP52c935
from pyrit.memory import SQLiteMemory

# Use in-memory database to avoid file corruption issues
memory = SQLiteMemory(db_path=":memory:")
memory.print_schema()

memory.dispose_engine()
