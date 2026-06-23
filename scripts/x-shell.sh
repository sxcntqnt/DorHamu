#!/bin/bash
set -e

# Install dependencies
apt-get update && apt-get install -y \
    wget \
    unzip \
    sumo \
    sumo-tools \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Install AStreet
wget https://github.com/a-b-street/abstreet/releases/download/v0.3.47/abstreet-linux-v0.3.47.zip
unzip abstreet-linux-v0.3.47.zip
mv abstreet /usr/local/bin/
rm abstreet-linux-v0.3.47.zip

# Install Cloverleaf
git clone https://github.com/Refefer/cloverleaf.git
cd cloverleaf
cargo build --release
# Do not remove the Cloverleaf directory
# cd .. 
# rm -rf cloverleaf

# Set SUMO environment
export SUMO_HOME=/usr/share/sumo

# Download OSM data for Nairobi
wget -O nairobi.osm "https://download.geofabrik.de/africa/kenya-latest.osm.pbf"
osmconvert nairobi.osm.pbf --out-osm > nairobi.osm

# Generate AStreet map
abstreet --import --osm nairobi.osm --output /tmp/nairobi.bin
gsutil cp /tmp/nairobi.bin gs://my_bucket/input_data/nairobi.bin

# Generate SUMO network
netconvert --osm-files nairobi.osm -o /tmp/nairobi.net.xml --geometry.remove --roundabouts.guess
polyconvert --osm-files nairobi.osm --net-file /tmp/nairobi.net.xml -o /tmp/nairobi.poly.xml
gsutil cp /tmp/nairobi.net.xml gs://my_bucket/input_data/nairobi.net.xml
gsutil cp /tmp/nairobi.poly.xml gs://my_bucket/input_data/nairobi.poly.xml

# Generate SUMO routes
cat <<EOF > /tmp/nairobi.rou.xml
<routes>
    <vType id="bus" vClass="bus" length="12" maxSpeed="16.7" accel="1.2" decel="4.5"/>
    <route id="route_0" edges="edge1 edge2 edge3"/>
</routes>
EOF

# Generate SUMO config
cat <<EOF > /tmp/nairobi.sumocfg
<configuration>
    <input>
        <net-file value="nairobi.net.xml"/>
        <route-files value="nairobi.rou.xml"/>
        <additional-files value="nairobi.poly.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="86400"/>
    </time>
    <report>
        <no-warnings value="true"/>
    </report>
</configuration>
EOF
gsutil cp /tmp/nairobi.sumocfg gs://my_bucket/input_data/nairobi.sumocfg
gsutil cp /tmp/nairobi.rou.xml gs://my_bucket/input_data/nairobi.rou.xml

echo "Setup completed successfully"
