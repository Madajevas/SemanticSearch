using System.Buffers;
using System.IO.Pipes;
using System.Runtime.CompilerServices;
using System.Text;

namespace Search
{
    internal sealed class EmbeddingClient : IDisposable
    {
        private NamedPipeClientStream client;

        public static async Task<EmbeddingClient> CreateAsync()
        {
            var embeddingsClient = new EmbeddingClient();
            await embeddingsClient.client.ConnectAsync();

            return embeddingsClient;
        }

        private EmbeddingClient()
        {
            client = new NamedPipeClientStream(".", "testpipe", PipeDirection.InOut);
        }

        public async IAsyncEnumerable<float> GetEmbeddingAsync(string text, [EnumeratorCancellation]CancellationToken cancellationToken)
        {
            var payload = Encoding.UTF8.GetBytes(text);
            await client.WriteAsync(BitConverter.GetBytes(payload.Length), cancellationToken);
            await client.WriteAsync(payload, 0, payload.Length, cancellationToken);

            var lengthBuffer = ArrayPool<byte>.Shared.Rent(sizeof(int));
            await client.ReadExactlyAsync(lengthBuffer, 0, sizeof(int), cancellationToken);
            var length = BitConverter.ToInt32(lengthBuffer, 0);
            ArrayPool<byte>.Shared.Return(lengthBuffer);
            for (int i = 0; i < length; i++)
            {
                var embeddingBuffer = ArrayPool<byte>.Shared.Rent(sizeof(float));
                await client.ReadExactlyAsync(embeddingBuffer, 0, sizeof(float), cancellationToken);
                var embeddingValue = BitConverter.ToSingle(embeddingBuffer, 0);
                ArrayPool<byte>.Shared.Return(embeddingBuffer);

                yield return embeddingValue;
            }
        }

        public void Dispose() => client.Dispose();
    }
}
