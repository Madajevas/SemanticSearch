using MessagePack;

using Microsoft.Extensions.Hosting;
using Microsoft.ML;

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
                //.Append(mlContext.Transforms.Text.ApplyWordEmbedding(outputColumnName: nameof(TransformedTextData.Vector), inputColumnName: "Tokens"))
                // Invoke-RestMethod -Uri https://mlpublicassets.blob.core.windows.net/assets/wordvectors/wiki.en.vec -OutFile ./wiki.en.vec
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding(
                    outputColumnName: nameof(TransformedTextData.Vector),
                    inputColumnName: "Tokens",
                    customModelFile: @"C:\projects\wiki.en.vec"));

            var textTransformer = textPipeline.Fit(emptyDataView);
            predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(textTransformer);

            return base.StartAsync(cancellationToken);
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            using var server = new NamedPipeServerStream("testpipe", PipeDirection.InOut);
            await server.WaitForConnectionAsync(stoppingToken);

            using var reader = new StreamReader(server);
            var text = await reader.ReadLineAsync(stoppingToken);
            var data = new TextData { Text = text };

            var embeddedText = predictionEngine.Predict(data);
            await MessagePackSerializer.SerializeAsync<float[]>(server, embeddedText.Vector);
            await server.FlushAsync(stoppingToken);
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
