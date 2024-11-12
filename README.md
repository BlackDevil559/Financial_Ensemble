## Ensemble Model for stock price predictions 

Team member: Bhumesh Gaur and Satyam Patil
### RNN (Recurrent Neural Networks)
RNNs are designed to handle sequential data, making them a natural fit for financial time-series forecasting. In financial data, RNNs can learn from past observations to predict future market behavior, such as stock prices or currency exchange rates. The network maintains a hidden state, allowing it to retain information from previous time steps. However, RNNs often struggle with long-term dependencies due to issues like vanishing gradients.

### GRU (Gated Recurrent Units)
GRUs are a variation of RNNs that aim to solve the problem of vanishing gradients by introducing gating mechanisms. They are more computationally efficient than traditional RNNs while still being capable of capturing long-range dependencies in financial data. GRUs are particularly effective in financial forecasting tasks like predicting market movements, asset prices, or even economic indicators, as they can selectively remember or forget information from past time steps.

### CNN (Convolutional Neural Networks)
CNNs, typically used in image processing, can also be applied to financial data, especially when working with sequential or time-series data. By using 1D convolutions, CNNs can detect local patterns or trends within financial time-series data (such as stock prices, market indices, etc.). They can capture short-term dependencies effectively and are particularly useful when working with highly volatile financial markets, where identifying specific features such as price spikes or trends is critical.

### LSTM (Long Short-Term Memory)
LSTMs are a type of RNN designed to better capture long-term dependencies in time-series data. They incorporate a memory cell that can store information over long periods, making them especially effective in predicting financial trends or forecasting stock prices. LSTMs are often preferred in financial data analysis because they can model complex patterns in time-series data, such as seasonality, trends, and cyclic behavior, while avoiding the vanishing gradient problem faced by standard RNNs.

### TimesNet
TimesNet is a deep learning model specifically designed for time-series forecasting. It utilizes multiple layers and architectures to effectively capture temporal dependencies in data. For financial forecasting, TimesNet can be highly effective in predicting stock prices, market trends, and economic data, as it can learn complex patterns and handle large-scale time-series data efficiently. Its versatility and ability to model long-range dependencies make it an ideal choice for many financial prediction tasks.



### 1. Test dataset on classic models: CNN, RNN, GRU, LSTM, and TimesNet
Run `run.py` in the folder corresponding to each model
e.g. 
```
cd path/CNN-for-Time-Series-Prediction
python run.py
```

Results and trained models will be stored at corresponding folders e.g. `CNN-for-Time-Series-Prediction/saved_models`, `CNN-for-Time-Series-Prediction/test_result_50`(50 means using 50 csvs to test)
### 2. Evaluate by comparing their results
#### Running `integrate_result.py` in folder `dataset_test` 
This creates the `integrated_eval_data_{n}` folder (n represents number of csvs used in test) and folder `integrated_predicted_data_{n}`.

The `integrated_eval_data_{n}` folder stores mse, mae and r2 information, and the `integrated_predicted_data_{n}` folder stores the predicted result data.

#### Running `eval_output.py` in folder `dataset_test` 
This creates a table of comparisons based on the `integrated_eval_data_{n}` folder created before e.g. `merged_eval_data_5.csv`

### For example
Using CNN to test our dataset:

1. run `CNN-for-Time-Series-Prediction/run.py` to use our dataset to train CNN


| Folders             | Sub folders                                                | 
|---------------------|------------------------------------------------------------|
| test_result_5       | AMD_nonsentiment_2024013123, AMD_sentiment_2024013123, ... |
| test_result_25      | AMD_nonsentiment_2024013123, AMD_sentiment_2024013123, ... |
| test_result_50      | AMD_nonsentiment_2024013123, AMD_sentiment_2024013123, ... |
   
   you may get result like this

2. run `integrate_result.py` to integrate data

| Metric          | GRU           | CNN           | LSTM          | RNN           | TimesNet      |
|-----------------|---------------|---------------|---------------|---------------|---------------|
| MAE             | 0.013692475   | 0.024043111   | 0.014487029   | 0.030090562   | 0.028123611   |
| MSE             | 0.000354704   | 0.000897477   | 0.000406081   | 0.001951202   | 0.001378664   |
| R2              | 0.920132321   | 0.797917934   | 0.908563921   | 0.560653936   | 0.679685415   |
| Stock_symbol    | KO            | KO            | KO            | KO            | KO            |

   you may get integrated result like this

3. run `eval_output.py` to get `merged_eval_data_{n}.csv` to compare results from diffrent models
![Screenshot 2024-11-13 034934](https://github.com/user-attachments/assets/4e7ac2d4-17f8-4d52-8e8d-fae2b7ee1d52)
![Screenshot 2024-11-13 034914](https://github.com/user-attachments/assets/0a895cb1-4848-4476-ac2b-d7ada376a1c6)
![Screenshot 2024-11-13 034858](https://github.com/user-attachments/assets/8ac4969b-a5ec-43ed-bbe1-8ed5850605e0)
![Screenshot 2024-11-13 034842](https://github.com/user-attachments/assets/07b2bf7b-f8f8-4150-8a63-d2da94a6652d)
![Screenshot 2024-11-13 034816](https://github.com/user-attachments/assets/09f47514-3bd2-4129-a707-bb268a372a66)
![Screenshot 2024-11-13 034759](https://github.com/user-attachments/assets/39ac7622-6e42-4645-8c47-a51f640d9453)
![Screenshot 2024-11-13 034744](https://github.com/user-attachments/assets/05085973-c65d-442b-ab91-eb318b015a3f)
![Screenshot 2024-11-13 034731](https://github.com/user-attachments/assets/ac356490-270f-40be-85eb-91030591ef94)
![Screenshot 2024-11-13 034714](https://github.com/user-attachments/assets/98060aa3-403c-4647-9982-6b1ed9dbaccb)
![Screenshot 2024-11-13 034658](https://github.com/user-attachments/assets/5be64601-a112-4f85-ab4c-348c28ae327e)
![Screenshot 2024-11-13 034643](https://github.com/user-attachments/assets/271055c8-5c25-4eb4-8deb-2a57cd3693b3)
![Screenshot 2024-11-13 034626](https://github.com/user-attachments/assets/4914c0f7-e0d5-445a-8d98-42dc75b13e90)
![Screenshot 2024-11-13 034606](https://github.com/user-attachments/assets/fabbac00-2704-40c9-a9a1-831ae98ef470)
![Screenshot 2024-11-13 034542](https://github.com/user-attachments/assets/ccebb394-73c2-4417-a231-fad9e279baac)
![Screenshot 2024-11-13 034517](https://github.com/user-attachments/assets/7a277d5f-4fc8-490b-a8dd-8c0d1247f01b)




References

[1] https://github.com/Zdong104/FNSPID_Financial_News_Dataset

[2] https://arxiv.org/abs/2402.06698

[3] http://www.yahoofinance.com/

[4] https://github.com/thuml/TimesNet



