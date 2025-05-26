using MessagePack;

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

            await MessagePackSerializer.SerializeAsync<float[]>(server, [1, 2, 3]);
            await server.FlushAsync(stoppingToken);
        }
    }
}
