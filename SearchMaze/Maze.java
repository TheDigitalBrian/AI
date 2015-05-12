package edu.uab.cis.search.maze;

import java.util.Set;

/**
 * A rectangular maze consisting of a start square, a goal square and some
 * obstacle squares between them.
 */
public class Maze {

  private int rows;
  private int columns;
  private Square start;
  private Square goal;
  private Set<Square> obstacles;

  /**
   * Creates a rectangular maze.
   * 
   * @param rows
   *          The number of rows of squares in the maze.
   * @param columns
   *          The number of columns of squares in the maze.
   * @param start
   *          The start square, where the maze begins.
   * @param goal
   *          The goal square, where the maze ends.
   * @param obstacles
   *          Squares within the maze that cannot be traveled through.
   */
  public Maze(int rows, int columns, Square start, Square goal, Set<Square> obstacles) {
    this.rows = rows;
    this.columns = columns;
    this.start = start;
    this.goal = goal;
    this.obstacles = obstacles;
    if (this.isBlocked(start)) {
      throw new IllegalArgumentException("Start square is blocked");
    }
    if (this.isBlocked(goal)) {
      throw new IllegalArgumentException("Goal square is blocked");
    }
  }

  /**
   * @return The start square, where the maze begins.
   */
  public Square getStart() {
    return this.start;
  }

  /**
   * @return The goal square, where the maze ends.
   */
  public Square getGoal() {
    return this.goal;
  }

  /**
   * Determines if the given square is blocked by an obstacle.
   * 
   * @param square
   *          A square from the maze to be tested.
   * @return True if the square represents an obstacle (or is out of bounds).
   */
  public boolean isBlocked(Square square) {
    int row = square.getRow();
    int column = square.getColumn();
    return row < 0 || row >= this.rows || column < 0 || column >= this.columns
        || this.obstacles.contains(square);
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    // first row of obstacles
    for (int column = 0; column < this.columns + 2; ++column) {
      builder.append("#");
    }
    builder.append("\n");
    // main rows of maze
    for (int row = 0; row < this.rows; ++row) {
      builder.append("#");
      for (int column = 0; column < this.columns; ++column) {
        Square square = new Square(row, column);
        if (square.equals(this.getStart())) {
          builder.append('S');
        } else if (square.equals(this.getGoal())) {
          builder.append('G');
        } else if (this.obstacles.contains(square)) {
          builder.append('#');
        } else {
          builder.append(' ');
        }
      }
      builder.append("#\n");
    }
    // final row of obstacles
    for (int column = 0; column < this.columns + 2; ++column) {
      builder.append("#");
    }
    builder.append("\n");
    // create the resulting string
    return builder.toString();
  }
}
