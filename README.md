This project develops an Early Warning and Decision Support System (EWDSS) for Aadhaar enrolment and update operations by transforming large-scale administrative data into actionable operational intelligence. Instead of relying on raw monthly volumes, the system analyzes how enrolment activity behaves over time and across geography, enabling proactive intervention rather than reactive firefighting.
The framework monitors enrolment patterns at district and pincode levels, decomposing operational stress into four complementary signals: volatility (chronic month-to-month unpredictability), spike frequency (recurring abnormal surges), peak stress (capacity shocks in extreme months), and severity (operational impact of abnormal events). These signals are integrated into a single Instability Score, providing a clear, comparable measure of operational risk across regions. The outputs are also showcased on a locally hosted Dashboard.
To support early warning, the system employs robust anomaly detection to flag unusual activity before it escalates into service congestion. Severity scoring further prioritizes anomalies by real-world impact, ensuring attention is directed to events that matter operationally. Concentration analysis (Top-X contribution and inequality measures) highlights whether instability is localized in a small number of regions or systemic, allowing targeted allocation of resources.
Finally, unsupervised clustering groups districts and pincodes into interpretable instability types (e.g., stable, spiky, peaky, mixed), which are mapped to reason codes and corresponding operational responses. This converts complex analytics into decision-ready insights—such as where to deploy surge staffing, activate early alerts, adjust seasonal capacity, or conduct root-cause investigations.
Overall, the proposed EWDSS enables UIDAI to anticipate enrolment stress, prioritize high-risk regions, and optimize operational planning, strengthening service reliability while improving efficiency and responsiveness of Aadhaar enrolment operations.

Key Decision Outcomes

•	Proactive capacity planning using month-ahead forecasts for both district and pincodes.
•	Risk-based monitoring and audits using anomaly rankings, instability scores, and reason codes.
•	Cluster-based governance: common playbooks for regions with similar behaviour (scalable policy design).
•	Stable benchmarks via consistent-performer identification.

Problem Statement

Build a system that enables UIDAI to proactively monitor Aadhaar enrolment activity, detect abnormal patterns early, explain what is happening, and forecast future workload to support evidence-based operational decisions.

Innovation & Novelty

1.	Explainability-first scoring: transparent calculations and reason codes for administrators.
2.	Composite instability index combining coefficient of variation, spike frequency, peak behaviour and severity into a single operational risk score.
3.	Cluster-based governance using K-Means to segment regions by behaviour for scalable policy design.
4.	Lightweight EWMA forecasting for actionable month-ahead planning with uncertainty bands.
5.	A dashboard hosted locally for a quick glimpse of outputs.

Pipeline Overview

1.	Parse date and derive month.
2.	Identify Age columns and converting it to numeric.
3.	Monthly aggregation at district and pincode levels.
4.	Spike metrics and severity scoring (abnormal month-to-month changes).
5.	Top anomalies by month (ranked alerts).
6.	Instability scoring and reason codes (explainability layer).
7.	Top X contribution in instability
8.	Clustering of regions by behavioral similarity using K-Means.
9.	Preparing reason code (Volatile, Spiky, Peaky, Mixed).
10.	EWMA forecasting for next-month workload.
11.	Identification of consistent performers (stable benchmarks).
12.	Locally hosted Dashboard through Streamlit.
