# GNU Makefile
#
# This file is part of the LibTiePie programming examples.
#
# Find more information on http://www.tiepie.com/LibTiePie .

CC = gcc
CFLAGS = -O2 -I. -Wall
LD = gcc
LFLAGS = -ltiepie

ifeq ($(OS),Windows_NT)
  CFLAGS += -std=c99
  LFLAGS += -L./
  TARGET_EXT = .exe
  RM = del
else
  CFLAGS += -std=gnu99
  LFLAGS += -lm
  TARGET_EXT =
  RM = rm -f
endif

SOURCES = $(wildcard Generator*.c) \
          $(wildcard Oscilloscope*.c) \
          $(wildcard I2C*.c) \
          ListDevices.c \
          load.c \
          gradient_descent.c \
          smith.c \
          pressure_block_position.c \
          marker_stream.c \
          demodulation_stream.c \
          marker_stream_demod.c \
          ati_stream_df10.c \
          pressure_block_pchange.c \
          mouse_stream.c \
          signal_test_stream.c \
          stream.c
       

DEPENDENCIES = CheckStatus.c \
               PrintInfo.c \
               Utils.c  \
               npy.c

OBJECTS = $(SOURCES:.c=.o)
DEPOBJECTS = $(DEPENDENCIES:.c=.o)

TARGETS = $(SOURCES:.c=$(TARGET_EXT))

.PHONY : all clean

all : $(TARGETS)

clean :
	$(RM) $(DEPOBJECTS) $(OBJECTS) $(TARGETS)

%.o : %.c
	$(CC) $(CFLAGS) $< -c -o $@

$(TARGETS) : %$(TARGET_EXT) : $(DEPOBJECTS) %.o
	$(LD) $+ -o $@ $(LFLAGS)
