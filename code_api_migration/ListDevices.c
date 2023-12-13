/**
 * ListDevices.c
 *
 * This example prints all the available devices to the screen.
 *
 * Find more information on https://www.tiepie.com/libtiepie-hw-sdk .
 */

#include "CheckStatus.h"
#include "PrintInfo.h"
#include "Utils.h"
#include <inttypes.h>
#include <libtiepie-hw.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
  int status = EXIT_SUCCESS;

  // Initialize library:
  tiepie_hw_init();

  // Print library information:
  printLibraryInfo();

  // Enable network search:
  tiepie_hw_network_set_auto_detect_enabled(TIEPIE_HW_BOOL_TRUE);
  CHECK_LAST_STATUS();

  // Update device list:
  tiepie_hw_devicelist_update();
  CHECK_LAST_STATUS();

  // Get the number of connected devices:
  const uint32_t connectedDevices = tiepie_hw_devicelist_get_count();
  CHECK_LAST_STATUS();

  if(connectedDevices != 0)
  {
    printf("\nAvailable devices:\n");

    for(uint32_t index = 0; index < connectedDevices; index++)
    {
      const tiepie_hw_handle item = tiepie_hw_devicelist_get_item_by_index(index);

      // Print device info:
      uint32_t length = tiepie_hw_devicelistitem_get_name(item, NULL, 0) + 1;
      CHECK_LAST_STATUS();
      char* name = malloc(sizeof(char) * length);
      length = tiepie_hw_devicelistitem_get_name(item, name, length);
      CHECK_LAST_STATUS();
      printf("  Name: %s\n", name);
      free(name);

      printf("    Serial Number  : %" PRIu32 "\n", tiepie_hw_devicelistitem_get_serial_number(item));
      CHECK_LAST_STATUS();

      printf("    Available types: ");
      printDeviceType(tiepie_hw_devicelistitem_get_types(item));
      CHECK_LAST_STATUS();
      printf("\n");

      if(tiepie_hw_devicelistitem_has_server(item))
      {
        tiepie_hw_handle server = tiepie_hw_devicelistitem_get_server(item);
        CHECK_LAST_STATUS();

        length = tiepie_hw_server_get_url(server, NULL, 0) + 1;
        CHECK_LAST_STATUS();
        char* url = malloc(sizeof(char) * length);
        length = tiepie_hw_server_get_url(server, url, length);
        CHECK_LAST_STATUS();

        length = tiepie_hw_server_get_name(server, NULL, 0) + 1;
        CHECK_LAST_STATUS();
        name = malloc(sizeof(char) * length);
        length = tiepie_hw_server_get_name(server, name, length);
        CHECK_LAST_STATUS();

        printf("    Server         : %s (%s)\n", url, name);

        free(url);
        free(name);

        printServerInfo(server);
        tiepie_hw_object_close(server);
        CHECK_LAST_STATUS();
      }

      tiepie_hw_object_close(item);
    }
  }
  else
  {
    fprintf(stderr, "No devices found!\n");
  }

  // Exit library:
  tiepie_hw_fini();

  return status;
}
