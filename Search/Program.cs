
using CsvHelper;
using CsvHelper.Configuration;

using Dapper;

using Microsoft.Data.SqlClient;

using Search;

using System.Diagnostics;
using System.Globalization;
using System.Linq;

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

var sw = new Stopwatch();
sw.Start();
uint processed = 0;

using var embeddingClient = new EmbeddingClient();
using var moviesFileStream  = new StreamReader(@"C:\projects\imdb-genres\imdb_genres.csv");
using (var csv = new CsvReader(moviesFileStream, new CsvConfiguration(CultureInfo.InvariantCulture)))
{
    csv.Context.RegisterClassMap<MovieMap>();
    foreach (var movie in csv.GetRecords<Movie>())
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        var embedding = await embeddingClient.GetEmbeddingAsync(movie.Description, cts.Token);

        var sql = $"""
            INSERT INTO movies (name, description, vector)
            VALUES (@Title, @Description, CAST(JSON_ARRAY({string.Join(',', embedding)}) AS VECTOR(900)))
            """;
        await connection.ExecuteAsync(sql, movie);

        if (++processed % 1000 == 0)
        {
            Console.WriteLine($"Processed {processed} movies in {sw.ElapsedMilliseconds} ms");
        }
    }
}

Console.WriteLine("Done! Press any key to exit.");
Console.ReadKey();

record Movie
{
    public string Title { get; init; }
    public string Description { get; init; }
}

class MovieMap : ClassMap<Movie>
{
    public MovieMap()
    {
        Map(m => m.Title).Name("movie title - year");
        Map(m => m.Description).Name("description");
    }
}
