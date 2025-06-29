
using CsvHelper;
using CsvHelper.Configuration;

using Dapper;

using EmbeddingsService.Client;

using Microsoft.Data.SqlClient;

using System.Data.Common;
using System.Diagnostics;
using System.Globalization;

bool isTableCreated = false;
var connectionString = "Server=127.0.0.1,1433;Database=main;User Id=sa;Password=test-1234;Database=movies;MultipleActiveResultSets=True;TrustServerCertificate=true;";
using var connection = new SqlConnection(connectionString);
await connection.OpenAsync();


var sw = new Stopwatch();
sw.Start();
uint processed = 0;

using var embeddingClient = new EmbeddingClient();
// Invoke-RestMethod -uri https://www.kaggle.com/api/v1/datasets/download/ashpalsingh1525/imdb-movies-dataset -OutFile C:\temp\imdb-movies-dataset.zip
// Expand-Archive -Path C:/temp/imdb-movies-dataset.zip -DestinationPath C:/temp/
using var moviesFileStream  = new StreamReader(@"C:\temp\imdb_movies.csv");
using (var csv = new CsvReader(moviesFileStream, new CsvConfiguration(CultureInfo.InvariantCulture)))
{
    csv.Context.RegisterClassMap<MovieMap>();
    foreach (var movieChunk in csv.GetRecords<Movie>().DistinctBy(m => m.Title).Chunk(500))
    {
        using var transaction = await connection.BeginTransactionAsync();

        foreach (var movie in movieChunk)
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            var embedding = await embeddingClient.GetEmbeddingAsync(movie.Description, cts.Token);
            await EnsureMoviesTable(connection, transaction, (ushort)embedding.Length);

            var sql = $"""
            INSERT INTO movies (name, description, vector)
            VALUES (@Title, @Description, CAST(JSON_ARRAY({string.Join(',', embedding)}) AS VECTOR({embedding.Length})))
            """;
            await connection.ExecuteAsync(sql, movie, transaction);

            processed++;
        }

        await transaction.CommitAsync();
        Console.WriteLine($"Processed {processed} movies in {sw.Elapsed}");
    }
}

Console.WriteLine("Done! Press any key to exit.");
Console.ReadKey();

async ValueTask EnsureMoviesTable(SqlConnection connection, DbTransaction transaction, ushort vectorLength)
{
    if (isTableCreated)
    {
        return;
    }

    await connection.ExecuteAsync($"""
        IF  NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[movies]') AND type in (N'U'))
        BEGIN
        CREATE TABLE movies (
            id INT IDENTITY (1,1),
            name VARCHAR(500) NOT NULL,
            description TEXT NOT NULL,
            vector VECTOR({vectorLength}) NOT NULL
        );
        END
        """, transaction: transaction);
    isTableCreated = true;
}

record Movie
{
    public string Title { get; init; }
    public string Description { get; init; }
}

class MovieMap : ClassMap<Movie>
{
    public MovieMap()
    {
        Map(m => m.Title).Name("names");
        Map(m => m.Description).Name("overview");
    }
}
