1. Business Understanding
    -What business problem are we trying to solve?
    -Why do we need ML instead of business rules?
    -Who will use the predictions?
    -Who are the stakeholders?
    -What happens today without the model?
    -What pain are investigators currently facing?
    -How much money do we lose every month?

2. Data Understanding
    -What does one row represent?
    -What is the prediction target?
    -Are there duplicate transactions?
    -What does Time represent?
    -Are all transactions included?
    -Are there missing features?
    -Are some features unavailable at prediction time?

3. Label Understanding
    -Who labels fraud?
    -Is it customer-reported?
    -Is it investigator-confirmed?
    -Can labels change later?
    -How long does it take before a transaction is confirmed as fraud?
    -Are labels always correct?
    -Can one transaction have multiple outcomes?

4. Success Metrics
    -What business KPI should improve?
    -Is missing fraud worse than blocking legitimate customers?
    -What is the acceptable false positive rate?
    -What recall do we need?
    -Should we optimize for Precision or Recall?
    -How many alerts can investigators review daily?

5. Production Requirements
    -Is prediction real-time or batch?
    -What is the maximum acceptable prediction latency?
    -How often will the model retrain?
    -What happens if the model is unavailable?
    -Can investigators provide feedback?
    -How is the model integrated into existing systems?

6. Risks & Compliance
    -What happens if we miss fraud?
    -What happens if we wrongly block a customer?
    -Are there legal consequences?
    -Are there compliance requirements?
    -Is explainability required?
    -Can the model be audited?

7. Assumptions
    -Can we use customer age?
    -Can we use location?
    -Can we use previous fraud history?
    -Are there privacy restrictions?
    -How much historical data is available?
    -What's the compute budget?
    -Do predictions need to be made within milliseconds?
8. Open Questions