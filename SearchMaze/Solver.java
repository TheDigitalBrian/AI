package edu.uab.cis.search.maze;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import com.google.common.collect.Sets;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.PriorityQueue;

/**
 * Solves a maze using A* search with an L1 heuristic.
 *
 * Specifically, squares are explored with the following strategy: <ul>
 * <li>Squares are ordered for exploration using the score f(x) = g(x) + h(x),
 * with the smallest f(x) squares being explored first</li> <li>g(x) is the
 * length of the path so far, from the start square to the current square,
 * including the steps necessary to avoid obstacles</li> <li>h(x) is the L1
 * estimate of the path to the goal, that is, the Manhattan distance to the
 * goal, ignoring potential obstacles</li> <li>Squares with the same f(x) score
 * are ordered by the h(x) score, with smaller h(x) scores first</li>
 * <li>Squares with the same f(x) and h(x) scores are ordered by row, with
 * smaller rows first</li> <li>Squares with the same f(x), h(x) and row should
 * be ordered by column, with smaller columns first</li> </ul>
 */
public class Solver {

    private Set<Square> explored;
    private List<Square> path;
    private HashMap<Square, Square> parentSquare = new HashMap<Square, Square>();
    private Square start;
    private Square goal;

    /**
     * Solves the given maze, determining the path to the goal.
     *
     * @param maze The maze to be solved.
     */
    public Solver(Maze maze) {
        path = new ArrayList<Square>();
        explored = Sets.newHashSet();
        start = maze.getStart();
        goal = maze.getGoal();

        Square current = start;
        Square childSquare;

        int currentColumn = start.getColumn();
        int currentRow = start.getRow();

        Comparator<Square> comp = new AStarComparator(maze);
        // 11 is default size for priority queue
        PriorityQueue<Square> queue = new PriorityQueue<Square>(11, comp);

        // Loop until goal square is reached.
        while (!current.equals(goal)) {
            // Check to see if each neighbor, or child square, of the current
            // square is valid. If so, add to queue.
            childSquare = new Square(currentRow - 1, currentColumn);
            if (isValidSquare(maze, childSquare, queue) == true) {
                parentSquare.put(childSquare, current);
                queue.add(childSquare);
            }

            childSquare = new Square(currentRow + 1, currentColumn);
            if (isValidSquare(maze, childSquare, queue) == true) {
                parentSquare.put(childSquare, current);
                queue.add(childSquare);
            }

            childSquare = new Square(currentRow, currentColumn - 1);
            if (isValidSquare(maze, childSquare, queue) == true) {
                parentSquare.put(childSquare, current);
                queue.add(childSquare);
            }

            childSquare = new Square(currentRow, currentColumn + 1);
            if (isValidSquare(maze, childSquare, queue) == true) {
                parentSquare.put(childSquare, current);
                queue.add(childSquare);
            }

            // If the queue runs out of squares, then there is no solution.
            if (queue.size() != 0) {
                // Go to the next square in the queue based on a star algorithm.
                current = queue.poll();
                currentColumn = current.getColumn();
                currentRow = current.getRow();

                // If the goal state has not been reached, explore current square.
                // If the goal state has been reached, get the path from the start
                // square to the goal square.
                if (!current.equals(goal)) {
                    explored.add(current);
                } else {
                    path = getPathFromSquare(current);
                }
            } else {
                System.out.println("There is no solution.");
                break;
            }
        }
    }

    /**
     * @return The squares along the path from the start to the goal, not
     * including the start square and the goal square.
     */
    public List<Square> getPathFromStartToGoal() {
        return this.path;
    }

    /**
     * @return All squares that were explored during the search process. This is
     * always a superset of the squares returned by
     * {@link #getPathFromStartToGoal()}.
     */
    public Set<Square> getExploredSquares() {
        return this.explored;
    }

    /**
     * Obtains the path from the start square to the given square.
     *
     * @param square A square from the maze.
     * @return All squares that form the path to the given square from the start
     * square.
     */
    private List<Square> getPathFromSquare(Square s) {
        List<Square> pathFromSquare = new ArrayList<Square>();
        Square parent = parentSquare.get(s);
        Square child = s;

        // Backtrack to each parent square until start is reached.
        while (!parent.equals(start)) {
            pathFromSquare.add(parent);
            if (!parent.equals(start)) {
                child = parent;
                parent = parentSquare.get(child);
            }
        }
        
        // Reverse the list so the path order is correct.
        Collections.reverse(pathFromSquare);
        return pathFromSquare;
    }

    /**
     * Determines if the given square should be added to the priority queue.
     *
     * @param m The maze that is to be solved.
     * @param s One of the neighbors of the current square.
     * @param queue The priority queue that contains all squares that have been
     * reached but not visited.
     * @return True if the square is not blocked on the maze, not part of the 
     * queue, not the start state, and not in the explored squares set.
     */
    private boolean isValidSquare(Maze m, Square s, PriorityQueue<Square> queue) {
        if (m.isBlocked(s) != true && explored.contains(s) != true
                && !s.equals(start) && queue.contains(s) != true) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * @param s A square from the maze whose manhattan distance is desired.
     * @param goal The square the maze is trying to reach.
     * @return The manhattan distance from the given square to the goal square.
     */
    private int getManhattanDistance(Square s, Square goal) {
        return Math.abs(s.getColumn() - goal.getColumn())
                + Math.abs(s.getRow() - goal.getRow());
    }

    /**
     * A comparator to be used by the priority queue that prioritizes squares
     * based on the heuristic and distance traveled.
     */
    private class AStarComparator implements Comparator<Square> {

        Maze m;
        Square g;

        /**
         * Creates a comparator based on the maze.
         *
         * @param m The maze that is to be solved.
         */
        public AStarComparator(Maze m) {
            this.m = m;
            this.g = m.getGoal();
        }

        /**
         * @param s A square in the maze that is part of the queue.
         * @return The sum of the manhattan distance and the distance traveled
         * by the square.
         */
        private int aStarFunction(Square s) {
            Square goal = m.getGoal();
            int manhattanDistance = getManhattanDistance(s, goal);
            int exploredSize = getPathFromSquare(s).size() + 1;
            return manhattanDistance + exploredSize;
        }

        /**
         * @param x A square in the maze that is part of the queue.
         * @param y A second square in the maze that is part of the queue.
         * @return The value that represents the priority of x over y.
         */
        @Override
        public int compare(Square x, Square y) {
            if (aStarFunction(x) < aStarFunction(y)) {
                return -1;
            }
            if (aStarFunction(x) > aStarFunction(y)) {
                return 1;
            }
            // tiebreaker one: manhattan distance
            if (getManhattanDistance(x, g) < getManhattanDistance(y, g)) {
                return -1;
            }
            if (getManhattanDistance(x, g) > getManhattanDistance(y, g)) {
                return 1;
            }
            // tiebreaker two: row value
            if (x.getRow() < y.getRow()) {
                return -1;
            }
            if (x.getRow() > y.getRow()) {
                return 1;
            }
            // tiebreaker three: column value
            if (x.getColumn() < y.getColumn()) {
                return -1;
            }
            if (x.getColumn() > y.getColumn()) {
                return 1;
            }
            // THIS SHOULD NOT HAPPEN
            return 0;
        }
    }
}
