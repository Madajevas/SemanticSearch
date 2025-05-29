
using Dapper;

using MessagePack;

using Microsoft.Data.SqlClient;

using System.IO.Pipes;

// docker volume create mssql_vector
// docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=test-1234" -p 1433:1433 --user root -v mssql_vector:/var/opt/mssql/data -d mcr.microsoft.com/mssql/server:2025-latest

var connectionString = "Server=127.0.0.1,1433;Database=main;User Id=sa;Password=test-1234;Database=movies;MultipleActiveResultSets=True;TrustServerCertificate=true;";
using var connection = new SqlConnection(connectionString);
await connection.ExecuteAsync("""
    IF  NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[movies]') AND type in (N'U'))
    BEGIN
    CREATE TABLE movies (
        id INT IDENTITY (1,1),
        name VARCHAR(500) NOT NULL,
        description TEXT NOT NULL,
        vector VECTOR(900) NOT NULL
    );
    END
    """);



using var client = new NamedPipeClientStream(".", "testpipe", PipeDirection.InOut);
await client.ConnectAsync();

using var writer = new StreamWriter(client);
writer.AutoFlush = true;
using var reader = new StreamReader(client);

writer.WriteLine("Hello, Server!");

using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
var embedding = await MessagePackSerializer.DeserializeAsync<float[]>(client, cancellationToken: cts.Token);

Console.WriteLine("Message sent to the server. Press any key to exit.");