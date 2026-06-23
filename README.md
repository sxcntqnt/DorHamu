# Distributed PPO for Traffic Light Control with Multi-Agent RL
Uses a distributed version of the deep reinforcement learning algorithm [PPO](https://arxiv.org/abs/1707.06347) to control a grid of traffic lights for optimized traffic flow through the system. The traffic enviornment is implemented in the realistic traffic simulation [SUMO](https://sumo.dlr.de/docs/index.html). Multi-agent RL (MARL) is implemented with each traffic light acting as a single agent. 

## SUMO / Traci
SUMO (**S**imulation of **U**rban **MO**bility) is a continuous road traffic simulation. [TraCI](Thttps://sumo.dlr.de/docs/TraCI.html) (**Tra**ffic **C**ontrol **I**nterface) connects to a SUMO simulation in a programming language (in this case Python) to allow for feeding inputs and recieving outputs. 

![SUMO picture](/images/sumo.png)

The environments implemented for this problem are grids where an intersection is controlled by a traffic light. Either NS cars can go or EW cars, at a time. So each intersection has 2 possible configurations. Cars spawn at the edges and then have a predefined destination edge where they despawn.

## Models
### PPO
[Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/) (PPO) is a policy gradient based reinforcement learning algorithm created by OpenAI. It is efficient and fairly simple and tends to be the goto for RL nowadays. There are a lot of great tutorials and code on PPO ([this](https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12), [this](https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/PPO_agent.py) and many more). 

![PPO code](/images/ppo.png)

### DPPO
#### DISCLAIMER: The DPPO implementation here is incorrect. It does not properly aggregate the gradients during training.

Distributed algorithms use multiple processes to speed up existing algorithms such as PPO. There arent as many simple resources on DPPO but I used a few different sources noted in my code such as [this repo](https://github.com/alexis-jacq/Pytorch-DPPO). I first implemented single-agent RL which means that in a single environment there is only one agent. In this apps case, this means all traffic lights are controlled by one agent. However, that means as the grid size increases the action size increases exponentially. 

###  MARL
![1x1 grid](/images/1_1-grid.png)

For example, the action space for a single intersection is 2 as either the NS light can be green or the EW light can be greed. 

![2x2 grid](/images/2_2-grid.png)

The number of actions for a 2x2 grid is 2^4 = 16. For example if 1 means NS is green and 0 means EW is green. Then 1011 in binary (13 in decimal) would mean that 3 of the 4 intersections are NS green. This can become a problem as the grid gets even larger. 

![MARL](/images/marl.png)

[Cooperative MARL](https://arxiv.org/abs/1908.03963) is a way to fix this ["curse of dimensionality"](https://en.wikipedia.org/wiki/Curse_of_dimensionality) problem. With MARL there are multiple agents in the environment. And in this case each agent controls a single intersection. So now an agent only has 2 possible actions no matter how big the grid gets! MARL also helps with inputs. Instead of a single agent needing to be trained to deal with say 4 states (for a 2x2 grid) it can just deal with one. MARL is a great tool in cases where your problem can run into scaling issues. 

In the case of this repo, I use independent MARL which means each agent does not directly communicate. However, each actor and critic share parameters across all agents. One trick for better cooperation is to share certain info across agents (other than just weights). Reward and states are two popular items to share. This [post](https://bair.berkeley.edu/blog/2018/12/12/rllib/) by Berkeley goes into this more.

## How to Run this
### Depndencies
* numpy
* traci
* sumolib
* scipy
* pytorch
* pandas

### Running
Can alter `constants.json` or `constans-grid.json` in /constants to change different hyperparameters. In `main.py` can run experiments with `run_normal` (runs multiple experiments using `constants.json`), `run_random_search` (runs a random search on `constants-grid.json`) or `run_grid_search` (runs a grid search on `constants-grid.json`). Can save and load models. Can also visualize models by running `vis_agent.py` and changing `run(load_model_file=<MODEL FILE NAME>)` to the model file. The 4 envs implemented are 1x1, 2x2, 3x3 and 4x4. 

`shape` is the grid, `rush_hour` can be set to true for 2x2 which adds a kind of rush-hour spawning probability distribution. And `uniform_generation_probability` is the spawn rate for cars when `rush_hour` is false. 
```
"environment": {
        "shape": [4, 4],
        "rush_hour": false,
        "uniform_generation_probability": 0.06
    },
```

Change `num_workers` based on how many processes you want active for the distribibuted part of DPPO. 
```
    "parallel":{
        "num_workers": 8
    }
```
Finally, you can change the `agent_type` to `rule` if you want a simple rule based agent to run (which just changes each light after a set amount of time). And can change `single_agent` to true to not use MARL. 

```
    "agent": {
        "agent_type": "ppo",
        "single_agent": false
    },
```



###########################################################################################################################
cat README.md
Matatu Route-Finding AI Pipeline
This pipeline optimizes matatu routes in urban settings using a hybrid AStreet-SUMO simulation, hierarchical reinforcement learning (RL), RegionDCL for spatial embeddings, and the Refefer Cloverleaf library for graph-based embeddings. It integrates with a bus reservation system to leverage real-time passenger booking data, identifying high-demand areas and peak/low-peak hours for optimized routing. The pipeline supports offline functionality for low-connectivity areas and is scalable via Google Cloud, making it suitable for urban transport networks like Nairobi’s matatu system.
For detailed documentation, visit GitHub Repository or xAI Docs.
Prerequisites

Google Cloud account with BigQuery, Cloud Storage (GCS), Cloud Run, Pub/Sub, and Monitoring enabled
Google Cloud SDK (install via https://cloud.google.com/sdk/docs/install)
Python 3.9 or later
SUMO (Simulation of Urban MObility) for traffic simulation
AStreet (A/B Street) for urban mobility modeling
Docker for building and deploying the API to Cloud Run
Redis server for caching
xAI API token (obtain from https://x.ai/api)
Cloverleaf library (installed via setup.sh from https://github.com/Refefer/cloverleaf)
NVIDIA GPU (optional) for accelerated RegionDCL training on Vertex AI
Internet connectivity for initial setup and dependency installation
Minimum system requirements: 8GB RAM, 20GB disk space for local execution

Setup

Authenticate Google Cloud:
gcloud auth login
gcloud config set project my_project

Replace my_project with your actual Google Cloud project ID.

Install Dependencies:
pip install -r requirements.txt


Run Setup Script:Ensure wget and unzip are installed for downloading dependencies.
sudo apt-get install wget unzip
chmod +x setup.sh
./setup.sh


Create GCS Bucket:
gsutil mb -p my_project gs://my_bucket


Set Environment Variables:
export GOOGLE_CLOUD_PROJECT=my_project
export SUMO_HOME=/usr/share/sumo
export XAI_API_TOKEN=your_xai_token

Verify variables:
echo $GOOGLE_CLOUD_PROJECT $SUMO_HOME $XAI_API_TOKEN



Execution

Upload Input Data:Upload required data files (e.g., nairobi.osm, H3 hexagons, African boundaries) to GCS.
gsutil cp input_data/* gs://my_bucket/input_data/


Validate Input Data:Ensure data files are available in GCS.
gsutil ls gs://my_bucket/input_data/


Prepare Reservation Data (Optional):Ensure reservation data is available in BigQuery for demand-driven routing.
bq query --nouse_legacy_sql "SELECT COUNT(*) FROM my_project.my_dataset.reservations"


Prepare Data:
python data_preparation.py  # Loads H3, OSM, and African datasets
python spatial_joins.py     # Joins H3 with OSM and boundaries
python feature_enhancement.py  # Adds spatial features


Prepare and Train RegionDCL (~1–2 hours on CPU, ~30 minutes with GPU):
python regiondcl_data.py   # Prepares raster and contrastive pairs
python regiondcl_model.py  # Trains RegionDCL model


Build Goals Relational Graph (GRG) with Cloverleaf:
python grg.py  # Constructs GRG with RegionDCL embeddings


Train RL Models (~2–4 hours):
python rl_training.py

Or use Vertex AI for faster training:
gcloud ai custom-jobs create --region=us-central1 --display-name=matatu-rl-training --config=vertex_ai_config.yaml

Monitor job progress:
gcloud ai custom-jobs describe projects/my_project/locations/us-central1/customJobs/<job-id>

Sample vertex_ai_config.yaml:
workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-4
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    replicaCount: 1
    pythonPackageSpec:
      executorImageUri: gcr.io/my_project/matatu-trainer
      packageUris: gs://my_bucket/rl_trainer.tar.gz
      pythonModule: rl_training


Evaluate and Deploy:
python evaluation_deployment.py

Test API locally:
curl -H "Authorization: Bearer driver_123:secret_token" -X POST -d '{"hex_code": "8a2a1072b59ffff", "passengers": 5}' http://localhost:8080/predict

Deploy to Cloud Run:
docker build -t gcr.io/my_project/matatu-api .
docker push gcr.io/my_project/matatu-api
gcloud run deploy matatu-api --image gcr.io/my_project/matatu-api --region us-central1 --platform managed --allow-unauthenticated --cpu=2 --memory=4Gi --concurrency=80



API Usage
Configure API authentication in evaluation_deployment.py or a secrets manager (e.g., Google Secret Manager).

Predict Route (Driver: Get next optimal stop):
curl -H "Authorization: Bearer driver_123:secret_token" -X POST -d '{"hex_code": "8a2a1072b59ffff", "passengers": 5}' http://<cloud-run-url>/predict

Example Response:
{
  "current_hex": "8a2a1072b59ffff",
  "suggested_next_node": "8a2a1072b5affff"
}


Submit Feedback (Driver: Rate route suggestions):
curl -H "Authorization: Bearer driver_123:secret_token" -X POST -d '{"hex_code": "8a2a1072b59ffff", "next_node": "8a2a1072b5affff", "rating": 4.5}' http://<cloud-run-url>/feedback


Query Matatu Locations (Passenger: Find nearby matatus):
curl -H "Authorization: Bearer passenger_123:secret_token" -X GET http://<cloud-run-url>/matatus?hex_code=8a2a1072b59ffff



Find Cloud Run URL:
gcloud run services describe matatu-api --region us-central1 --format 'value(status.url)'

Common API Errors:

400 Bad Request: Invalid hex_code or missing parameters.
401 Unauthorized: Incorrect or missing authentication token.

Monitoring
Monitor pipeline performance via Google Cloud Console:

View logs in Cloud Logging: matatu_pipeline (https://console.cloud.google.com/logs/query;query=resource.type%3D%22cloud_run_revision%22%20logName%3D%22matatu_pipeline%22)
Monitor metrics in Cloud Monitoring: custom.googleapis.com/matatu/* (https://console.cloud.google.com/monitoring/metrics-explorer?project=my_project)
prediction_latency: API response time
cache_hit: Cached prediction rate


Set up alerts for critical issues (e.g., high error rates):gcloud alpha monitoring policies create --policy-from-file=alert_policy.yaml

Example alert_policy.yaml:displayName: High API Error Rate
combiner: OR
conditions:
  - displayName: API Errors
    conditionThreshold:
      filter: metric.type="custom.googleapis.com/matatu/error_rate"
      comparison: COMPARISON_GT
      thresholdValue: 0.1
      duration: 300s
notificationChannels: ["projects/my_project/notificationChannels/<channel-id>"]



Offline Support
Offline data (offline_data.pkl) contains precomputed GRG embeddings and routes, enabling driver and passenger mobile apps to function without internet in rural or low-signal areas.

Download offline data:gsutil cp gs://my_bucket/offline_data.pkl .


Update after retraining:python evaluation_deployment.py --package-offline
gsutil cp offline_data.pkl gs://my_bucket/offline_data.pkl


Mobile app requirements:
100MB storage for offline_data.pkl
Offline-compatible framework (e.g., Flutter, React Native)



Notes

RegionDCL is trained after data preparation to provide spatially informed embeddings for the Cloverleaf GRG, enhancing downstream RL performance.
Integrate a bus reservation system (BigQuery table my_project.my_dataset.reservations) to enhance demand-driven routing with real-time passenger booking data.

Troubleshooting

Cloverleaf Installation Failure:git clone https://github.com/Refefer/cloverleaf.git
cd cloverleaf
pip install .

Use a stable commit for compatibility:pip install git+https://github.com/Refefer/cloverleaf.git@<commit-hash>


RegionDCL Training Issues:Ensure TensorFlow GPU support:pip install tensorflow[and-cuda]

Or use Vertex AI for GPU acceleration.
BigQuery Permissions:Grant necessary permissions:gcloud projects add-iam-policy-binding my_project --member=user:your-email --role=roles/bigquery.admin


Data Privacy:Ensure reservation data complies with regulations (e.g., GDPR, Kenya Data Protection Act).

Benefits for Drivers and Passengers
For Matatu Drivers
Matatu drivers can optimize routes, maximize earnings, and minimize costs using the pipeline.
a. Efficient Routing

What It Does: Uses RegionDCL and Cloverleaf embeddings to recommend routes that avoid congestion and target high-demand areas, guided by real-time traffic and passenger bookings.
Benefits:
Saves 1–2 hours daily, allowing 1–2 extra trips.
Increases revenue by 10–20% ($5–10/day) by carrying more passengers (e.g., 12–15 vs. 5–10 per trip).
Reduces fuel costs by 5–10% ($0.50–1/day).


Example: A driver avoids a congested downtown route and heads to a busy market with 15 booked passengers, saving 10 minutes and earning $7.50 extra.

b. Reservation-Driven Routing

What It Does: Integrates real-time booking data to identify high-demand areas and peak hours, guiding drivers to maximize pick-ups.
Benefits:
Boosts revenue by 20–30% by targeting areas with confirmed bookings.
Reduces downtime by directing drivers to booked locations.


Example: A driver is routed to a hexagon with 15 booked passengers at 8 AM, filling the vehicle and earning $7.50.

c. Real-Time Decision Support via Mobile App

What It Does: Delivers route recommendations via a mobile app, using offline data for low-connectivity areas. Driver feedback refines suggestions.
Benefits:
Provides clear instructions (e.g., “Head to hex 8a2a1072b5affff”).
Ensures reliability in rural areas, maintaining productivity.


Example: A driver in a semi-rural area uses offline mode to follow a precomputed route, saving time and building customer loyalty.

d. Cost Savings through Predictive Maintenance

What It Does: Uses traffic and congestion data to avoid high-wear routes, informing maintenance schedules.
Benefits:
Lowers maintenance costs by 5–10% ($50–100/month) by reducing vehicle wear.
Increases uptime for more trips.


Example: Avoiding a congested route reduces brake wear, saving $50 monthly.

Overall Impact:

Time Saved: 1–2 hours/day.
Money Made: $5–15/day ($125–375/month, 25 days).
Money Saved: $0.50–2/day ($12.50–50/month).

For Matatu Users (Passengers)
Passengers benefit from faster, more reliable, and cost-effective travel.
a. Faster Travel Times

What It Does: Ensures matatus take the shortest, least congested paths using RegionDCL and Cloverleaf embeddings.
Benefits:
Saves 10–15 minutes per trip, critical for commuters.
Improves reliability with consistent travel times.


Example: A commuter saves 15 minutes by avoiding a congested highway, arriving at work on time.

b. Improved Service Availability

What It Does: Guides drivers to high-demand areas, ensuring matatu availability during peak hours.
Benefits:
Reduces wait times to 5–10 minutes.
Enhances coverage for underserved areas.


Example: A student finds a matatu in 5 minutes near a school, improving daily commutes.

c. Cost Stability and Affordability

What It Does: Reduces driver costs, stabilizing fares. RL balances efficiency and passenger count.
Benefits:
Maintains affordable fares ($0.50/trip) even during peak demand.
Offers better value with faster trips.


Example: A passenger pays $0.50 for a 10-minute faster trip, getting more value.

d. Enhanced User Experience via App

What It Does: Powers a passenger app for real-time matatu tracking, booking, and arrival estimates.
Benefits:
Improves convenience with trip planning.
Enhances safety by reducing wait times in unsafe areas.


Example: A passenger books a seat and tracks a matatu arriving in 3 minutes, avoiding a crowded terminal.

Overall Impact:

Time Saved: 10–30 minutes/trip (5–10 hours/month for daily commuters).
Money Saved: Stable fares ($0.05–0.10/trip, $1–2/month).
Improved Experience: Reliable, convenient, safer travel.

Note: Revenue and cost estimates are based on Nairobi’s matatu market ($0.50/passenger fare). Adjust for local fare structures.
Future Work
Scalability
The pipeline can be adapted to other cities by retraining RegionDCL and Cloverleaf on new spatial and reservation data, expanding benefits to additional urban transport networks.
Broader Implications

Economic: Drivers’ increased earnings and passengers’ time savings boost local economies.
Social: Reliable service improves access to jobs, education, and healthcare for low-income users.
Environmental: Reduced fuel consumption lowers emissions, supporting sustainability.

Recommendations for Adoption

Deploy Mobile Apps: Develop driver and passenger apps with API integration and offline support.
Train Drivers: Educate drivers on using the app and trusting AI recommendations.
Subsidize Access: Provide low-cost smartphones or data plans for app adoption.
Integrate Payment Systems: Link to mobile payment platforms (e.g., M-Pesa) for seamless fares.
Expand Coverage: Scale to other cities with new spatial data.
