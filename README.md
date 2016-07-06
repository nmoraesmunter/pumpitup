# PUMP IT UP

I tried to follow CRISP-DM process

## Business understanding

- [Problem description](https://www.drivendata.org/competitions/7/page/25/)

## Data understanding

- [Getting data from](https://www.drivendata.org/competitions/7/data/)

## Data preparation

- I created dummies from categorical columns
- I created 3 new columns from the date column storing day of year, month and year
- A lot of columns contained redundant information, I kept the ones with more detailed information.
  I dropped the following columns
    ..* 'wpt_name'
    ..* 'basin'
    ..* 'subvillage'
    ..* 'region'
    ..* 'ward'
    ..* 'lga'
    ..* 'recorded_by'
    ..* 'scheme_name'
    ..* 'extraction_type_group'
    ..* 'extraction_type_class'
    ..* 'management_group'
    ..* 'payment_type'
    ..* 'quality_group'
    ..* 'quantity_group'
    ..* 'source_type'
    ..* 'source_class'
    ..* 'waterpoint_type_group'
    ..* 'funder'
    ..* 'installer'


  - TO-DO:
    ..* Explore outliers (latitude, longitude)
    ..* Ensure that dropped features are not important

## Modeling
  As my first approach I used Random Forest. May be interesting try AdaBoosting

## Evaluation

  The score obtained after second submission was: 0.8154

## Deployment

  --
