

using Dapper;

using EmbeddingsService.Client;

using Microsoft.Data.SqlClient;

using Spectre.Console;

var connectionString = "Server=127.0.0.1,1433;Database=main;User Id=sa;Password=test-1234;Database=movies;MultipleActiveResultSets=True;TrustServerCertificate=true;";
using var connection = new SqlConnection(connectionString);
await connection.OpenAsync();

var noOfDimensions = await connection.ExecuteScalarAsync<ushort>("""
    SELECT vector_dimensions FROM sys.columns AS c
    INNER JOIN sys.tables AS t ON t.object_id = c.object_id
    WHERE t.name = 'movies' and c.name = 'vector'
    """);

using var embeddingClient = new EmbeddingClient();

while (true)
{
    var phrase = AnsiConsole.Prompt(new TextPrompt<string>("Phrase to look for:"));
    var search = await embeddingClient.GetEmbeddingAsync(phrase, default);
    var query = $"""
    SELECT TOP (20) name, description, VECTOR_DISTANCE('cosine', CAST(JSON_ARRAY({string.Join(',', search)}) AS VECTOR({noOfDimensions})), [vector]) AS distance
    FROM movies
    ORDER BY distance
    """;
    var results = await connection.QueryAsync<(string, string, float)>(query);
    var table = new Table();
    table.AddColumn("Name");
    table.AddColumn("Description");
    table.AddColumn("Distance");
    foreach (var (name, description, distance) in results)
    {
        table.AddRow(name, description, distance.ToString());
    }
    AnsiConsole.Write(table);
}
