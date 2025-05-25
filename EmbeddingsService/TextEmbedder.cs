using Microsoft.Extensions.Hosting;

using System.IO.Pipes;

namespace EmbeddingsService
{
    internal class TextEmbedder : BackgroundService
    {
        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            using var server = new NamedPipeServerStream("testpipe", PipeDirection.InOut);
            await server.WaitForConnectionAsync(stoppingToken);

            using var reader = new StreamReader(server);
            var text = await reader.ReadLineAsync(stoppingToken);

            using var writer = new StreamWriter(server) { AutoFlush = true };
            writer.WriteLine($"Embedded: {text}");
        }
    }
}
