# Data Management and Visualization Practicals (DMVP)

This repository contains four Python projects covering data handling, preprocessing, exploratory analysis, API-based data collection, and visualization.

## Projects

| Practical | Project | Main concepts |
|---|---|---|
| DMVP 1 | Sales Data Analysis | CSV, Excel, JSON, Pandas, aggregation, Matplotlib |
| DMVP 2 | Weather Data Analysis | REST API, requests, JSON, time-series analysis, correlation |
| DMVP 3 | Telecom Customer Churn Analysis | Data cleaning, missing values, categorical analysis, IQR outlier detection |
| DMVP 4 | Real Estate Price Analysis | Data preprocessing, ZIP-code analysis, descriptive statistics, outlier detection |

## Requirements

Python 3.10 or later is recommended.

Install the required packages:

```bash
pip install pandas numpy matplotlib requests openpyxl
```

## Folder structure

```text
DMVP/
├── DMVP1_Sales_Analysis.py
├── DMVP2_Weather_Analysis.py
├── DMVP3_Telecom_Churn.py
├── DMVP4_RealEstate_Analysis.py
├── README.md
├── RealEstate_Prices.csv
├── Telecom_Customer_Churn.csv
└── outputs/
```

The CSV datasets for DMVP 3 and DMVP 4 must be placed in the repository root, or the file paths in the scripts must be updated.

DMVP 1 creates its own sample sales dataset. DMVP 2 retrieves forecast data from OpenWeatherMap.

## Run the projects

```bash
python DMVP1_Sales_Analysis.py
python DMVP2_Weather_Analysis.py
python DMVP3_Telecom_Churn.py
python DMVP4_RealEstate_Analysis.py
```

## Weather API key setup

Do not write your API key directly in the Python file or commit it to GitHub.

On Windows PowerShell, set the key for the current terminal session:

```powershell
$env:OPENWEATHER_API_KEY="YOUR_NEW_API_KEY"
python DMVP2_Weather_Analysis.py
```

If an API key was previously uploaded publicly, revoke or regenerate it in your OpenWeatherMap account.

## Notes on data processing

- Missing categorical values in the telecom dataset are labelled `Unknown`; this avoids assuming that missing service information means the customer did not use that service.
- The telecom binary churn label is created only for `Churned` and `Stayed`. `Joined` is not silently classified as `Stayed`.
- IQR outliers are identified and reported. They are not automatically removed because an unusual value may be a valid observation.
- The real-estate dataset is not one-hot encoded because the current practical performs exploratory analysis rather than fitting a machine-learning model.
- Weather rain probability is converted to percentage only for display. The source API represents it as a value between 0 and 1.

## Data and execution

The scripts expect the dataset filenames shown above. Check that you have permission to redistribute any dataset before committing it to a public repository.

The scripts have been reviewed and improved based on the submitted practical PDFs. Runtime verification requires the original CSV datasets and, for the weather project, a valid API key.
