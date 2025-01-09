// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <vector>
#include <map>
#include <algorithm>

#include "neml2/models/DependencyDefinition.h"
#include "neml2/misc/string_utils.h"

namespace neml2
{
/**
 * @brief The DependencyResolver identifies and resolves the dependencies among a set of objects
 * derived from DependencyDefinition.
 *
 * @tparam Node The type of the node in the dependency graph, i.e. if this resolver is used to
 * figure out dependencies among Models, this template parameter should be Model.
 * @tparam ItemType The type of the consumed/provided items of each node
 */
template <typename Node, typename ItemType>
class DependencyResolver
{
public:
  /**
   * Similar to @tparam ItemType but additionally contains information about its @tparam Node
   * parent, i.e. the node which defines this consumed/provided item.
   */
  struct Item
  {
    Item(Node * const node, ItemType item)
      : parent(node),
        value(std::move(item))
    {
    }

    /// Node which defines this item
    Node * const parent;

    /// The consumed/provided item
    const ItemType value;

    /// Test for equality between two items
    bool operator==(const Item & other) const
    {
      return parent == other.parent && value == other.value;
    }

    /// Test for inequality between two items
    bool operator!=(const Item & other) const
    {
      return parent != other.parent || value != other.value;
    }

    /// An arbitrary comparator so that items can be sorted (for consistency)
    bool operator<(const Item & other) const
    {
      return parent != other.parent ? (parent < other.parent) : (value < other.value);
    }
  };

  DependencyResolver() = default;

  /// Add a node (defining consumed/provided items) in the dependency graph
  void add_node(DependencyDefinition<ItemType> *);

  /// Add an additional outbound item that the dependency graph _provides_
  void add_additional_outbound_item(const ItemType & item);

  /// Set a node's priority, useful for resolving cyclic dependency
  void set_priority(DependencyDefinition<ItemType> *, size_t);

  /// Resolve nodal dependency and find an evaluation order
  void resolve();

  /// The resolved (nodal) evaluation order following which all consumed items of the current node
  const std::vector<Node *> & resolution() const { return _resolution; }

  /**
   * The item-item provider dictionary: key of the dictionary is the item of interest, and the value
   * of the dictionary is the set of items that _provide_ the item of interest.
   */
  const std::map<Item, std::set<Item>> & item_providers() const { return _item_provider_graph; }

  /**
   * The item-item consumer dictionary: key of the dictionary is the item of interest, and the value
   * of the dictionary is the set of items that _consume_ the item of interest.
   */
  const std::map<Item, std::set<Item>> & item_consumers() const { return _item_consumer_graph; }

  /**
   * The node-node provider dictionary: key of the dictionary is the node of interest, and the value
   * of the dictionary is the set of nodes that _provide_ the item of interest.
   */
  const std::map<Node *, std::set<Node *>> & node_providers() const { return _node_provider_graph; }

  /**
   * The node-node consumer dictionary: key of the dictionary is the node of interest, and the value
   * of the dictionary is the set of nodes that _consume_ the item of interest.
   */
  const std::map<Node *, std::set<Node *>> & node_consumers() const { return _node_consumer_graph; }

  /// End nodes which are not consumed by anyone else
  const std::set<Node *> & end_nodes() const { return _end_nodes; }

  /// The items provided by the overall dependency graph, i.e., the items that are not consumed by _any_ node.
  const std::set<Item> & outbound_items() const { return _out_items; }

  /// Start nodes which do not consume anyone else
  const std::set<Node *> & start_nodes() const { return _start_nodes; }

  /// The items consumed by the overall dependency graph, i.e., the items that are not provided by _any_ node.
  const std::set<Item> & inbound_items() const { return _in_items; }

  /// @returns a boolean flag controlling whether item provider should be unique
  bool & unique_item_provider() { return _unique_item_provider; }

  /// @returns a boolean flag controlling whether item consumer should be unique
  bool & unique_item_consumer() { return _unique_item_consumer; }

private:
  /// Build adjacency graphs, find start/end nodes, find in/outbound items
  void build_graph();

  /// Helper recursive method for resolve()
  void resolve(Node *);

  /// Should item providers be unique?
  bool _unique_item_provider = true;

  /// Should item consumers be unique?
  bool _unique_item_consumer = false;

  /// Nodes of the DAG
  std::set<Node *> _nodes;

  /// Consumed items
  std::set<Item> _consumed_items;

  /// Provided items
  std::set<Item> _provided_items;

  /// Dependencies among the items, e.g., the directed edges of the DAG
  /// This is also referred to as the adjacency matrix of a graph
  std::map<Item, std::set<Item>> _item_provider_graph;

  /// Symmetric version of _item_provider_graph, i.e. stores consumers of each node rather than providers
  /// of each node.
  std::map<Item, std::set<Item>> _item_consumer_graph;

  /// Dependencies among the nodes, e.g., the directed edges of the DAG
  /// This is also referred to as the adjacency matrix of a graph
  std::map<Node *, std::set<Node *>> _node_provider_graph;

  /// Symmetric version of _node_provider_graph, i.e. stores consumers of each node rather than providers
  /// of each node.
  std::map<Node *, std::set<Node *>> _node_consumer_graph;

  /// The end nodes, i.e. nodes that aren't needed by anyone else
  std::set<Node *> _end_nodes;

  /// The start nodes, i.e. nodes that don't depend on anyone else
  std::set<Node *> _start_nodes;

  /// The outbound items, i.e. items that aren't needed by anyone else
  std::set<Item> _out_items;

  /// The inbound items, i.e. items that don't depend on anyone else
  std::set<Item> _in_items;

  /// The resolved order to evaluate all the nodes
  std::vector<Node *> _resolution;

  /// Track visited nodes during BFS
  std::map<Node *, int> _status;

  /// Priority of nodes used to break cyclic dependency
  std::map<Node *, size_t> _priority;
};

template <typename Node, typename ItemType>
void
DependencyResolver<Node, ItemType>::add_node(DependencyDefinition<ItemType> * def)
{
  auto node = dynamic_cast<Node *>(def);
  _nodes.emplace(node);

  for (const auto & item : node->consumed_items())
    _consumed_items.emplace(node, item);

  for (const auto & item : node->provided_items())
    _provided_items.emplace(node, item);
}

template <typename Node, typename ItemType>
void
DependencyResolver<Node, ItemType>::add_additional_outbound_item(const ItemType & item)
{
  _consumed_items.emplace(nullptr, item);
}

template <typename Node, typename ItemType>
void
DependencyResolver<Node, ItemType>::set_priority(DependencyDefinition<ItemType> * def,
                                                 size_t priority)
{
  auto node = dynamic_cast<Node *>(def);
  _priority[node] = priority;
}

template <typename Node, typename ItemType>
void
DependencyResolver<Node, ItemType>::build_graph()
{
  // Clear the previous graph
  _item_provider_graph.clear();
  _item_consumer_graph.clear();
  _node_provider_graph.clear();
  _node_consumer_graph.clear();
  _start_nodes.clear();
  _end_nodes.clear();
  _in_items.clear();
  _out_items.clear();

  // Build the adjacency matrix for item providers and node providers
  for (const auto & itemi : _consumed_items)
  {
    std::vector<Item> providers;

    for (const auto & itemj : _provided_items)
    {
      // Match consumer with provider
      if (itemi.value != itemj.value)
        continue;

      // No self dependency
      if (itemi.parent == itemj.parent)
        continue;

      // Enforce priority
      if (_priority[itemi.parent] > _priority[itemj.parent])
        continue;

      providers.push_back(itemj);
    }

    // If the user asks for unique providers, we should error if multiple providers have been
    // found. Otherwise, just put the first provider into the graph.
    if (!providers.empty())
    {
      if (_unique_item_provider)
        if (providers.size() != 1)
          throw NEMLException("Multiple providers have been found for item " +
                              utils::stringify(itemi.value));
      _item_provider_graph[itemi].insert(providers[0]);
      if (itemi.parent)
        _node_provider_graph[itemi.parent].insert(providers[0].parent);
    }
  }

  // Build the adjacency matrix for item consumers
  for (const auto & itemi : _provided_items)
  {
    std::vector<Item> consumers;

    for (const auto & itemj : _consumed_items)
    {
      // Skip additional outbound item
      if (!itemj.parent)
        continue;

      // Match provider with consumer
      if (itemi.value != itemj.value)
        continue;

      // No self dependency
      if (itemi.parent == itemj.parent)
        continue;

      // Enforce priority
      if (_priority[itemi.parent] < _priority[itemj.parent])
        continue;

      consumers.push_back(itemj);
    }

    // If the user asks for unique consumers, we should error if multiple consumers have been
    // found. Otherwise, just put the first consumer into the graph.
    if (!consumers.empty())
    {
      if (_unique_item_consumer)
        if (consumers.size() != 1)
          throw NEMLException("Multiple consumers have been found for item " +
                              utils::stringify(itemi.value));
      _item_consumer_graph[itemi].insert(consumers[0]);
      _node_consumer_graph[itemi.parent].insert(consumers[0].parent);
    }
  }

  // Find start nodes
  for (const auto & node : _nodes)
    if (_node_provider_graph.count(node) == 0)
      _start_nodes.insert(node);

  // Find end nodes
  for (const auto & node : _nodes)
    if (_node_consumer_graph.count(node) == 0)
      _end_nodes.insert(node);

  // Find inbound items
  for (const auto & item : _consumed_items)
    if (_item_provider_graph.count(item) == 0)
      _in_items.insert(item);

  // Find outbound items
  for (const auto & item : _provided_items)
    if (_item_consumer_graph.count(item) == 0)
      _out_items.insert(item);

  // Additional outbound items
  for (const auto & item : _consumed_items)
    if (!item.parent)
    {
      if (!_item_provider_graph.count(item))
        throw NEMLException("Unable to find provider of the additional outbound item " +
                            utils::stringify(item.value));
      for (const auto & provider : _item_provider_graph[item])
      {
        _out_items.insert(provider);
        _end_nodes.insert(provider.parent);
      }
    }
}

template <typename Node, typename ItemType>
void
DependencyResolver<Node, ItemType>::resolve()
{
  build_graph();

  _status.clear();
  _resolution.clear();
  for (const auto & node : _end_nodes)
    if (!_status[node])
      resolve(node);

  // Make sure each node appears in the resolution once and only once
  for (const auto & node : _nodes)
  {
    auto count = std::count(_resolution.begin(), _resolution.end(), node);
    if (count == 0)
      throw NEMLException(
          "Each node must appear in the dependency resolution. Node " + node->name() +
          " is missing. This is an internal error -- consider filing a bug report.");
    if (count > 1)
      throw NEMLException(
          "Each node must appear in the dependency resolution once and only once. Node " +
          node->name() + " appeared " + std::to_string(count) +
          " times. This indicates cyclic dependency.");
  }
}

template <typename Node, typename ItemType>
void
DependencyResolver<Node, ItemType>::resolve(Node * node)
{
  // Mark the current node as visiting (so that we know there is circular dependency when a
  // "visiting" node is visited again).
  _status[node] += 1;

  // Recurse for all the dependent nodes
  if (_node_provider_graph.count(node))
    for (const auto & dep : _node_provider_graph[node])
    {
      // The dependent node must either be "not visited" or "visited".
      // If the dependent node is "being visited", there must be cyclic dependency.
      if (_status[dep] == 1)
        throw NEMLException(
            "While resolving dependency, two nodes '" + node->name() + "' and '" + dep->name() +
            "' have (possibly indirect) cyclic dependency. The cyclic dependency can be "
            "resolved by explicitly setting the node priorities.");

      if (!_status[dep])
        resolve(dep);
    }

  // At this point, all the dependent nodes must have been pushed into the resolution. It is
  // therefore safe to push the current node into the resolution.
  _resolution.push_back(node);

  // Finished visiting this node
  _status[node] += 1;
}
} // namespace neml2
