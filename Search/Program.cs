
using MessagePack;

using Microsoft.Extensions.Hosting;

using System.IO.Pipes;

var builder = Host.CreateApplicationBuilder(args);
builder.AddServiceDefaults();

var app = builder.Build();


using var client = new NamedPipeClientStream(".", "testpipe", PipeDirection.InOut);
await client.ConnectAsync();

using var writer = new StreamWriter(client);
writer.AutoFlush = true;
using var reader = new StreamReader(client);

writer.WriteLine("Hello, Server!");

using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
var embedding = await MessagePackSerializer.DeserializeAsync<float[]>(client, cancellationToken: cts.Token);

Console.WriteLine("Message sent to the server. Press any key to exit.");