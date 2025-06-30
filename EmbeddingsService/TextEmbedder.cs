using FastBertTokenizer;

using Microsoft.Extensions.Hosting;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using System.Buffers;
using System.IO.Pipes;

namespace EmbeddingsService
{
    internal class TextEmbedder : BackgroundService
    {
        private const int InputLength = 128;

        private InferenceSession session = null!;
        private BertTokenizer tokenizer = null!;

        public override void Dispose()
        {
            session?.Dispose();
            base.Dispose();
        }

        public override Task StartAsync(CancellationToken cancellationToken)
        {
            session = new InferenceSession(@"C:\projects\e5-large-v2\onnx\model.onnx");
            tokenizer = new BertTokenizer();
            using var tokenJsonStream = File.OpenRead(@"C:\projects\e5-large-v2\tokenizer.json");
            tokenizer.LoadTokenizerJson(tokenJsonStream);

            return base.StartAsync(cancellationToken);
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    using var server = new NamedPipeServerStream("text_embedding", PipeDirection.InOut);
                    await server.WaitForConnectionAsync(stoppingToken);

                    while (server.IsConnected)
                    {
                        var lengthBuffer = ArrayPool<byte>.Shared.Rent(sizeof(int));
                        await server.ReadExactlyAsync(lengthBuffer, 0, sizeof(int), stoppingToken);
                        var length = BitConverter.ToInt32(lengthBuffer, 0);
                        ArrayPool<byte>.Shared.Return(lengthBuffer);

                        var payloadBuffer = ArrayPool<byte>.Shared.Rent(length);
                        await server.ReadExactlyAsync(payloadBuffer, 0, length, stoppingToken);
                        var text = System.Text.Encoding.UTF8.GetString(payloadBuffer, 0, length);
                        ArrayPool<byte>.Shared.Return(payloadBuffer);

                        var embeddedText = GetTextEmbedding(text);
                        await server.WriteAsync(BitConverter.GetBytes(embeddedText.Length), stoppingToken);

                        var bufferSize = sizeof(float) * embeddedText.Length;
                        var buffer = ArrayPool<byte>.Shared.Rent(bufferSize);
                        Buffer.BlockCopy(embeddedText, 0, buffer, 0, bufferSize);
                        await server.WriteAsync(buffer, 0, bufferSize, stoppingToken);
                        ArrayPool<byte>.Shared.Return(buffer);

                        await server.FlushAsync(stoppingToken);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error: {ex.Message}");
                }
            }
        }

        private float[] GetTextEmbedding(string text)
        {
            var (inputIds, attentionMask, tokenTypeIds) = tokenizer.Encode(text);
            attentionMask = Normalize(attentionMask);
            var inputIdsTensor = new DenseTensor<long>(Normalize(inputIds), [1, InputLength]);
            var attentionMaskTensor = new DenseTensor<long>(attentionMask, [1, InputLength]);
            var tokenTypeIdsTensor = new DenseTensor<long>(Normalize(tokenTypeIds), [1, InputLength]);
            var inputs = new List<NamedOnnxValue>
                        {
                            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                            NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor)
                        };

            // TODO: convert to ort api
            using var output = session.Run(inputs);

            var result = output.First().AsTensor<float>();
            return CalculateMean(result, attentionMask.Span);
        }

        private Memory<long> Normalize(Memory<long> input)
        {
            if (input.Length > InputLength)
            {
                return input[..InputLength];
            }

            else if (input.Length < InputLength)
            {
                var memory = (new long[InputLength]).AsMemory();
                input.CopyTo(memory);
                return memory;
            }

            return input;
        }

        private float[] CalculateMean(Tensor<float> result, ReadOnlySpan<long> attentionMask)
        {
            var inputLength = result.Dimensions[1];
            var hiddenLength = result.Dimensions[2];
            var meanEmbedding = new float[hiddenLength];
            var realTokenCount = 0;

            for (int i = 0; i < inputLength; i++)
            {
                if (attentionMask[i] == 1)
                {
                    for (int h = 0; h < hiddenLength; h++)
                    {
                        meanEmbedding[h] += result[0, i, h];
                    }
                    realTokenCount++;
                }
            }

            for (int h = 0; h < hiddenLength; h++)
            {
                meanEmbedding[h] /= realTokenCount;
            }

            return meanEmbedding;
        }
    }
}
