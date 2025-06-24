using System.Buffers;
using System.IO.Pipes;
using System.Text;

namespace EmbeddingsService.Client
{
    public sealed class EmbeddingClient : IDisposable
    {
        private NamedPipeClientStream client;

        public EmbeddingClient()
        {
            client = new NamedPipeClientStream(".", "text_embedding", PipeDirection.InOut, PipeOptions.Asynchronous);
        }

        public async ValueTask ConnectAsync()
        {
            if (client.IsConnected)
            {
                return;
            }

            await client.ConnectAsync();
        }

        public async Task<float[]> GetEmbeddingAsync(string text, CancellationToken cancellationToken)
        {
            await ConnectAsync();

            var payload = Encoding.UTF8.GetBytes(text);
            await client.WriteAsync(BitConverter.GetBytes(payload.Length), cancellationToken);
            await client.WriteAsync(payload, 0, payload.Length, cancellationToken);

            var lengthBuffer = ArrayPool<byte>.Shared.Rent(sizeof(int));
            await client.ReadExactlyAsync(lengthBuffer, 0, sizeof(int), cancellationToken);
            var length = BitConverter.ToInt32(lengthBuffer, 0);
            ArrayPool<byte>.Shared.Return(lengthBuffer);

            var embeddingBuffer = ArrayPool<byte>.Shared.Rent(sizeof(float) * length);
            await client.ReadExactlyAsync(embeddingBuffer, 0, sizeof(float) * length, cancellationToken);
            var vector = new float[length];
            Buffer.BlockCopy(embeddingBuffer, 0, vector, 0, sizeof(float) * length);
            ArrayPool<byte>.Shared.Return(embeddingBuffer);

            return vector;
        }

        public void Dispose() => client.Dispose();
    }
}
