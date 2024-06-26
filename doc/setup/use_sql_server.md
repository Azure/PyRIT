# Using PyRIT with Azure SQL Server

## Configure Azure SQL Server

In order to connect PyRIT with Azure SQL Server, an Azure SQL Server instance with username & password
authentication enabled is required. If you are creating a new Azure SQL Server resource, be sure to note the password for your "Server Admin." Otherwise, if you have an existing Azure SQL Server resource, you can reset the password from the "Overview" page.

PyRIT does not yet support Microsoft Entra ID (formerly known as Azure Active Directory) when accessing Azure SQL Server. Therefore, ensure your server is configured to take non-Entra connections. To do that, navigate in the Azure Portal to Settings -&gt; Microsoft Entra ID. Under the heading "Microsoft Entra authentication only", uncheck the box reading "Support only Microsoft Entra authentication for this server." Then save the configuration.

Finally, firewall rules can prevent you or your team from accessing SQL Server. To ensure you and your team have access, collect any public IP addresses of anyone who may need access to Azure SQL Server while running PyRIT. Once these are collected, navigate in the Azure Portal to Security -&gt; Networking. Under the heading "Firewall rules," click "+ Add a firewall rule" for each IP address that must be granted access. If the rule has only one IP address, copy the vame value into "Start IPv4 Address" and "End IPv4 Address." Then save this configuration.

## Configure SQL Database

Once you have created the server, ensure you have a database within the Azure SQL Server resource. You can create a new one by navigating in the Azure Portal to the "Overview" page and licking the "+ Create Database" button in the top menu and following the prompts.

## Configure Local Environment

Connecting PyRIT to an Azure SQL Server database requires ODBC, PyODBC and Microsoft's [ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver16) to be installed in your local environment. Consult PyODBC's [documentation](https://github.com/mkleehammer/pyodbc/wiki) for detailed instruction on.

## Connect PyRIT to Azure SQL Server Database

Once ODBC and the SQL Server driver have been configured, you must use the `AzureSQLMemory` implementation of `MemoryInterface` from the `pyrit.memory.azure_sql_server` module to connect PyRIT to an Azure SQL Server database.

The constructor for `AzureSQLMemory` requires a URL connection string of the form: `mssql+pyodbc://<username>:<password>@<serverName>.database.windows.net/<databaseName>?driver=<driver string>`, where `<username>` and `<password>` are the SQL Server username and password configured above, `<serverName>` is the "Server name" as specified on the Azure SQL Server "Overview" page, `<databaseName>` is the name of the database instance created above, and `<driver string>` is the driver identifier (likely `ODBC+Driver+18+for+SQL+Server` if you installed the latest version of Microsoft's ODBC driver).

## Use PyRIT with Azure SQL Server

Once all of the above steps are completed, you can connect to an Azure SQL Server database by invoking AzureSQLMemory. We recommend placing any secrets like your connection strings in a .env file and loading them, which the example shown below reflects.

```python
import os

from pyrit.common import default_values
from pyrit.memory import AzureSQLServer

default_values.load_default_env()

conn_str = os.environ.get('AZURE_SQL_SERVER_CONNECTION_STRING')

azure_memory = AzureSQLServer(connection_string=conn_str)
```

Once you have created an instance of `AzureSQLServer`, the code will ensure that your Azure SQL Server database is properly configured with the appropriate tables.
