# MILESTONE3

# HMM utilizing the avocado prices dataset
- Goal is to predict future avocado prices by training an HMM using the given dataset
## Raw Dataset

#### ([Dataset Link](https://www.kaggle.com/datasets/neuromusic/avocado-prices))
#### Historical data on avocado prices and sales volume in multiple US Markets

### Dataset Columns

- Date - The date of the observation
- AveragePrice - the average price of a single avocado
- type - conventional or organic
- year - the year
- Region - the city or region of the observation
- Total Volume - Total number of avocados sold
- 4046 - Total number of avocados with PLU 4046 sold
- 4225 - Total number of avocados with PLU 4225 sold
- 4770 - Total number of avocados with PLU 4770 sold


## Cleaned Dataset for training:

- Unnamed: 0 - index column added when the CSV was saved
- AveragePrice - The average price of avocados on a given day.
- Total Volume - The total volume of avocados sold on that day.
- 4046 - Volume of avocados sold with PLU 4046.
- 4225 - Volume of avocados sold with PLU 4225.
- 4770 - Volume of avocados sold with PLU 4770.
- Total Bags - Total number of bags sold, representing packaged quantities.
- Small Bags - Number of small-sized bags sold.
- Large Bags - Number of large-sized bags sold.
- XLarge Bags - Number of extra-large bags sold.
- type - Indicates the type of avocado sold (e.g., conventional or organic).
- year - The year when the sales were recorded.
- region - Specifies the geographical region or market where the sales took place.
- price_change - Represents the change in average price from a previous period.
- volume_change - Indicates the change in total volume sold compared to a previous period.
- price_moving_avg - The moving average of the avocado price over a set period, used to smooth out fluctuations.
- volume_moving_avg - The moving average of the sales volume, highlighting longer-term trends.
- price_state_encoded - An encoded version of the price data, used for modeling and to capture underlying price patterns.
- volume_state_encoded - An encoded version of the volume data, used for modeling and analysis.

## Variables and their interactions:
- AveragePrice – The key observed variable, representing the average selling price of avocados. It is influenced by supply, demand, and overall market conditions.
- Total Volume, 4046, 4225, 4770 – These variables represent the total sales volume and sales volumes for specific PLU codes. They capture the supply side and consumer purchasing behavior, which often correlate with price changes. For example, high volume might indicate abundant supply or a surge in demand.
- Total Bags, Small Bags, Large Bags, XLarge Bags – These packaging metrics add nuance by showing how products are sold in different formats. They can provide insights into market segmentation and trends in consumer preferences that indirectly affect pricing.
- type, year, region are Categorical variables that segment the data. Different regions or product types (e.g., conventional vs. organic) can experience distinct market dynamics, leading to variations in pricing and volume.
- price_change and volume_change – Derived variables that capture short-term dynamics. They indicate momentum or abrupt shifts that might signal transitions between market regimes.
- price_moving_avg and volume_moving_avg – These smooth out short-term fluctuations and highlight underlying trends. They serve as benchmarks to compare against instantaneous values, which helps in understanding whether current movements are anomalies or part of a longer trend.
- price_state_encoded and volume_state_encoded – These represent discretized or transformed versions of the continuous price and volume data. They can help in segmenting the observations into different states, facilitating the mapping to underlying hidden regimes.

## Why HMM
- The data in the avacodos set is sequential, and interdependent with its market data. This allows us to get accurate estimations utilizing inferring shifts in underlying market conditions.

## HMM structure
- Markov Chain:
  - The hidden state sequence follows a Markov process, meaning that the state at time t only depends on the state at time t-1

- Transition Matrix with Priors:
  - In this Bayesian framework, we place a prior on the transition probabilities. This allows us to incorporate prior beliefs and to quantify uncertainty in how states change over time.

- Emission Distributions:
  - For each hidden state, the observed variables are generated from a likelihood function parameterized by state-specific parameters. This involves:
  - A joint distribution that models the multivariate observation vector.
  - A factorized model where each variable is conditionally independent given the state.
- Priors on Emission Parameters:
- By placing priors on the emission parameters we capture uncertainty about the typical values of the variables when in a specific state. This enables full posterior inference over both the hidden states and the model parameters.



# Libraries Used:
- Used LabelEncoder, which is a tool in the scikit-learn library that is used to convert categorical values into a numerical form by assigning unique integers for each distinct category.
  - [Source](https://scikit-learn.org/stable/)