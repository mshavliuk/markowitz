Current program allows to estimate the [Sharp ratio](https://www.investopedia.com/terms/s/sharperatio.asp) for the given stocks by using the [Markowitz algorithm](https://en.wikipedia.org/wiki/Modern_portfolio_theory)

### Setup
1. clone project
```shell script
git clone https://github.com/mshavliuk/markowitz.git
```
2. Project relies on [direnv](https://direnv.net/) files to automate python virtualenv creation, but you can use your main python executable environment and set env vars manually.
```shell script
cd markowitz
direnv allow
direnv reload
```
3. Install dependencies
```shell script
pip3 install -r requirements.txt
```
4. Run the script
```shell script
python3 markowitz.py
```

![Example plot](example-plot.png?raw=true "Example plot")