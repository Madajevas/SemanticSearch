# How to
> Disregard aspire projects. I've shelved those for now.

* `docker volume create mssql_vector`
* `docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=test-1234" -p 1433:1433 --user root -v mssql_vector:/var/opt/mssql/data -d mcr.microsoft.com/mssql/server:2025-latest`
* Download https://huggingface.co/intfloat/e5-large-v2 and configure its location in `EmbeddingsService`
* Start `EmbeddingsService` project
* Download the dataset from https://www.kaggle.com/api/v1/datasets/download/ashpalsingh1525/imdb-movies-dataset and configure it in `Search.DataLoader` project
* Start `Search.DataLoader`
* Once data is loadd, start `Search.Search` project
