var builder = DistributedApplication.CreateBuilder(args);

var embeddings = builder.AddProject<Projects.EmbeddingsService>("embeddings");

var sql = builder.AddSqlServer("sql")
    .AddDatabase("semanticsearch");

var search = builder.AddProject<Projects.Search>("search")
    .WithReference(sql)
    .WithReference(embeddings);

builder.Build().Run();
