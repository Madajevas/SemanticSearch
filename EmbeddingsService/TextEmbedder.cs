using Microsoft.Extensions.Hosting;
using Microsoft.ML;

using System.Buffers;
using System.IO.Pipes;

namespace EmbeddingsService
{
    internal class TextEmbedder : BackgroundService
    {
        private PredictionEngine<TextData, TransformedTextData> predictionEngine;

        public override Task StartAsync(CancellationToken cancellationToken)
        {
            var mlContext = new MLContext();
            var emptyDataView = mlContext.Data.LoadFromEnumerable(Array.Empty<TextData>());

            var textPipeline = mlContext.Transforms.Text.NormalizeText(nameof(TextData.Text))
                .Append(mlContext.Transforms.Text.TokenizeIntoWords(outputColumnName: "Tokens", inputColumnName: nameof(TextData.Text)))
                // .Append(mlContext.Transforms.Text.ApplyWordEmbedding(outputColumnName: nameof(TransformedTextData.Vector), inputColumnName: "Tokens"))
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding(
                   outputColumnName: nameof(TransformedTextData.Vector),
                   inputColumnName: "Tokens",

                   // Invoke-RestMethod -Uri https://mlpublicassets.blob.core.windows.net/assets/wordvectors/wiki.en.vec -OutFile C:/temp/wiki.en.vec
                   // customModelFile: @"C:\projects\wiki.en.vec"

                   // Invoke-RestMethod -Uri https://nlp.stanford.edu/data/glove.840B.300d.zip -OutFile C:/temp/glove.840B.300d.zip
                   // Expand-Archive -Path C:/temp/glove.840B.300d.zip -DestinationPath C:/temp/glove.840B.300d
                   customModelFile: @"C:\projects\glove.840B.300d.txt"
                ));

            var textTransformer = textPipeline.Fit(emptyDataView);
            predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(textTransformer);

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

                        var data = new TextData { Text = text };
                        var embeddedText = predictionEngine.Predict(data);
                        await server.WriteAsync(BitConverter.GetBytes(embeddedText.Vector.Length), stoppingToken);

                        var bufferSize = sizeof(float) * embeddedText.Vector.Length;
                        var buffer = ArrayPool<byte>.Shared.Rent(bufferSize);
                        Buffer.BlockCopy(embeddedText.Vector, 0, buffer, 0, bufferSize);
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
    }

    class TextData
    {
        public string Text { get; set; }
    }

    class TransformedTextData : TextData
    {
        public float[] Vector { get; set; }
    }
}
