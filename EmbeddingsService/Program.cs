
using EmbeddingsService;

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = Host.CreateApplicationBuilder(args);
builder.AddServiceDefaults();
builder.Services.AddHostedService<TextEmbedder>();

var app = builder.Build();
// app.MapDefaultEndpoints();
await app.RunAsync();
