#[derive(Debug, Clone, Copy)]
pub struct NodeId(usize);

enum Node {
    Float(f64),
    Sum(NodeId, NodeId),
}

#[derive(Default)]
pub struct Context {
    nodes: Vec<Node>,
}

impl Context {
    fn new_node(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        NodeId(id)
    }

    fn get_node(&self, node: NodeId) -> &Node {
        self.nodes.get(node.0).expect("node id must exist")
    }

    fn get_node_mut(&mut self, node: NodeId) -> &mut Node {
        self.nodes.get_mut(node.0).expect("node id must exist")
    }

    pub fn set(&mut self, node: NodeId, new_value: f64) {
        let Node::Float(value) = self.get_node_mut(node) else {
            panic!("node must be a float");
        };
        *value = new_value;
    }

    pub fn param(&mut self, value: f64) -> NodeId {
        self.new_node(Node::Float(value))
    }

    pub fn sum(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.new_node(Node::Sum(a, b))
    }

    pub fn compute(&self, node: NodeId) -> f64 {
        match self.get_node(node) {
            Node::Float(node) => *node,
            Node::Sum(a, b) => self.compute(*a) + self.compute(*b),
        }
    }
}
