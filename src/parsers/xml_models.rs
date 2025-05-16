use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename = "network")] // Matches the root tag name
pub struct Network {
    #[serde(rename = "@version")]
    pub version: String,
    // xmlns is tricky with serde-xml-rs for the root element.
    // It's often implicitly handled if not specifically needed in the struct.
    // If you need to capture it:
    // #[serde(rename = "@xmlns")]
    // pub xmlns: String,
    #[serde(rename = "networkStructure")]
    pub network_structure: NetworkStructure,
    pub demands: Demands,
}

// <networkStructure> element
#[derive(Debug, Deserialize)]
pub struct NetworkStructure {
    pub nodes: Nodes,
    pub links: Links,
}

// <nodes> element
#[derive(Debug, Deserialize)]
pub struct Nodes {
    #[serde(rename = "@coordinatesType")]
    pub coordinates_type: String,
    #[serde(rename = "node", default)] // List of <node> elements
    pub node_list: Vec<Node>,
}

// <node> element
#[derive(Debug, Deserialize)]
pub struct Node {
    #[serde(rename = "@id")]
    pub id: String,
    pub coordinates: Coordinates,
}

// <coordinates> element
#[derive(Debug, Deserialize)]
pub struct Coordinates {
    pub x: f64,
    pub y: f64,
}

// <links> element
#[derive(Debug, Deserialize)]
pub struct Links {
    #[serde(rename = "link", default)] // List of <link> elements
    pub link_list: Vec<Link>,
}

// <link> element
#[derive(Debug, Deserialize)]
pub struct Link {
    #[serde(rename = "@id")]
    pub id: String,
    pub source: String,
    pub target: String,
    #[serde(rename = "preInstalledModule")]
    pub pre_installed_module: Option<PreInstalledModule>,
    #[serde(rename = "additionalModules")]
    pub additional_modules: AdditionalModules,
}

// <preInstalledModule> element
#[derive(Debug, Deserialize)]
pub struct PreInstalledModule {
    pub capacity: f64,
    pub cost: f64,
}

// <additionalModules> element
#[derive(Debug, Deserialize)]
pub struct AdditionalModules {
    #[serde(rename = "addModule", default)] // List of <addModule> elements
    pub add_module_list: Vec<AddModule>,
}

// <addModule> element
#[derive(Debug, Deserialize)]
pub struct AddModule {
    pub capacity: f64,
    pub cost: f64,
}

// <demands> element
#[derive(Debug, Deserialize)]
pub struct Demands {
    #[serde(rename = "demand", default)] // List of <demand> elements
    pub demand_list: Vec<Demand>,
}

// <demand> element
#[derive(Debug, Deserialize)]
pub struct Demand {
    #[serde(rename = "@id")]
    pub id: String,
    pub source: String,
    pub target: String,
    #[serde(rename = "demandValue")]
    pub demand_value: f64,
}
