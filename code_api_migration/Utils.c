/**
 * Utils.c
 *
 * This file is part of the LibTiePie programming examples.
 *
 * Find more information on https://www.tiepie.com/libtiepie-hw-sdk .
 */

#include "Utils.h"
#ifdef OS_WINDOWS
#  include <conio.h>
#  include <windows.h>
#else // POSIX
#  include <stdio.h>
#  include <termios.h>
#  include <unistd.h>
#endif

void sleepMilliSeconds(unsigned int ms)
{
#ifdef OS_WINDOWS
  Sleep(ms);
#else // POSIX
  usleep(ms * 1000);
#endif
}

void waitForKeyStroke()
{
#ifdef OS_WINDOWS
  getch();
#else // POSIX
  static struct termios old, new;

  // Backup old settings:
  tcgetattr(STDIN_FILENO, &old);

  new = old;              // Copy current settings to new.
  new.c_lflag &= ~ECHO;   // Disable echo mode.
  new.c_lflag &= ~ICANON; // Disable buffered I/O.

  // Set the new settings:
  tcsetattr(STDIN_FILENO, TCSANOW, &new);

  // Wait for key-press:
  getchar();

  // Restore old terminal I/O settings:
  tcsetattr(STDIN_FILENO, TCSANOW, &old);
#endif
}
