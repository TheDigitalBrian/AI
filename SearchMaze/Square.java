package edu.uab.cis.search.maze;

import java.util.Objects;

/**
 * A square in a maze, identified by a row and a column.
 */
public class Square {

  private int row, column;

  /**
   * Identifies a square on the maze via the given row and column.
   * 
   * @param row
   *          The row in the maze.
   * @param column
   *          The column in the maze.
   */
  public Square(int row, int column) {
    super();
    this.row = row;
    this.column = column;
  }

  /**
   * @return The row of this square in the maze.
   */
  public int getRow() {
    return this.row;
  }

  /**
   * @return The column of this square in the maze.
   */
  public int getColumn() {
    return this.column;
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.row, this.column);
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof Square) {
      Square that = (Square) obj;
      return this.row == that.row && this.column == that.column;
    }
    return false;
  }

  @Override
  public String toString() {
    return String.format("(%d,%d)", this.row, this.column);
  }

}
