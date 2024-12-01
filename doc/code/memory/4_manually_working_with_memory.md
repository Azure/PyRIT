# 4. Updating Memory Manually

After or during an operation or a test, it can be important to use the memory in the database in a straightforward way.

There are many ways to do this, but this section gives some general ideas on how users can solve common problems. Most of this relies on using https://duckdb.org/docs/guides/sql_editors/dbeaver.html

## Sharing Data Between Users

Eventually, we have plans to extend the `MemoryInterface` implementation to other instances. For example, it would not be a huge task to extend it to Azure SQL Server, and operators could use that as a shared database.

In the meantime, one of the easiest ways to share data is to do the following:

1. Export and import the database as described here. This allows a lot of flexibility and can include partial exports (for example based on labels or time):  https://dbeaver.com/docs/dbeaver/Data-transfer/
2. Copy the PyRIT `results/dbdata` directory over; it will contain multi-modal data that the database references.

## Making Pretty Graphs with Excel

This is especially nice with scoring. There are countless ways to do this, but this shows one example;

1. Do a query with the data you want. This is an example where we're only querying scores in the "float_type" scores in the category of "misinformation"

![scoring_1.png](../../../assets/scoring_1.png)

2. Export the data to a CSV

![scoring_2.png](../../../assets/scoring_2_export.png)

3. Use it as an excel sheet! You can use pivot tables, etc. to visualize the data.

![scoring_2.png](../../../assets/scoring_3_pivot.png)

4. Optionally, if you catch entries you want to update (e.g., if you want to correct scores), you could either change them in the database or in Excel and re-import. Note: the most stable way of doing this is in the database since the mapping can be off in some cases when reimporting.

## Entering Manual Prompts

Although most prompts are run through `PromptTargets` which will add prompts to memory, there are a few reasons you may want to enter in manual prompts. For example, if you ssh into a box, are not using PyRIT to probe for weaknesses, but want to add prompts later for reporting or scoring.

One of the easiest way to add prompts is through the `TextTarget` target. You can create a csv of prompts that looks as follows:

```
role, value
user, hello
assistant, hi how are you?
user, new conversation
```

This very simple format doesn't have very much information, but already it standardizes the prompts that can then be used in mass scoring (or manual scoring with HITLScorer).

And you can import it using code like this

```
target = TextTarget()
target.import_scores_from_csv(csv_file_path=".\path.csv")
```
