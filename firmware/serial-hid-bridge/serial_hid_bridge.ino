#include "Arduino.h"
#include "SwitchControllerPico.h"

const uint8_t desc_hid_report[] = {
  CUSTOM_DESCRIPTOR
};

Adafruit_USBD_HID usb_hid(
  desc_hid_report,
  sizeof(desc_hid_report),
  HID_ITF_PROTOCOL_NONE,
  2,
  false
);

namespace {

constexpr unsigned long BAUD_RATE = 115200;
constexpr uint16_t BUTTON_PRESS_MS = 100;
constexpr uint16_t COMMAND_GAP_MS = 20;
constexpr size_t MAX_COMMAND_LENGTH = 32;
constexpr int UART_TX_PIN = 0;  // GP0: Pico -> USB serial adapter RX
constexpr int UART_RX_PIN = 1;  // GP1: USB serial adapter TX -> Pico

String pending_command;

String normalizeCommand(String command) {
  command.trim();
  command.toUpperCase();
  return command;
}

bool sendButtonCommand(const String& command) {
  if (command == "A") {
    pushButton(Button::A, BUTTON_PRESS_MS, 1);
  } else if (command == "B") {
    pushButton(Button::B, BUTTON_PRESS_MS, 1);
  } else if (command == "X") {
    pushButton(Button::X, BUTTON_PRESS_MS, 1);
  } else if (command == "Y") {
    pushButton(Button::Y, BUTTON_PRESS_MS, 1);
  } else if (command == "START") {
    pushButton(Button::PLUS, BUTTON_PRESS_MS, 1);
  } else if (command == "SELECT") {
    pushButton(Button::MINUS, BUTTON_PRESS_MS, 1);
  } else {
    return false;
  }

  return true;
}

bool sendHatCommand(const String& command) {
  if (command == "UP") {
    pushHatButton(Hat::UP, BUTTON_PRESS_MS, 1);
  } else if (command == "DOWN") {
    pushHatButton(Hat::DOWN, BUTTON_PRESS_MS, 1);
  } else if (command == "LEFT") {
    pushHatButton(Hat::LEFT, BUTTON_PRESS_MS, 1);
  } else if (command == "RIGHT") {
    pushHatButton(Hat::RIGHT, BUTTON_PRESS_MS, 1);
  } else {
    return false;
  }

  return true;
}

bool executeCommand(const String& command) {
  if (command == "WAIT") {
    delay(BUTTON_PRESS_MS);
    return true;
  }

  if (command == "A_UNTIL_END_OF_DIALOG") {
    for (int i = 0; i < 6; ++i) {
      pushButton(Button::A, BUTTON_PRESS_MS, 1);
      delay(COMMAND_GAP_MS);
    }
    return true;
  }

  return sendButtonCommand(command) || sendHatCommand(command);
}

void handleCommand(String raw_command) {
  const String command = normalizeCommand(raw_command);
  if (command.length() == 0) {
    return;
  }

  if (executeCommand(command)) {
    switchcontrollerpico_reset();
    Serial1.print("OK ");
    Serial1.println(command);
  } else {
    switchcontrollerpico_reset();
    Serial1.print("ERR ");
    Serial1.println(command);
  }
}

void readSerialCommands() {
  while (Serial1.available() > 0) {
    const char c = static_cast<char>(Serial1.read());

    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      handleCommand(pending_command);
      pending_command = "";
      continue;
    }

    if (pending_command.length() < MAX_COMMAND_LENGTH) {
      pending_command += c;
    } else {
      pending_command = "";
      switchcontrollerpico_reset();
      Serial1.println("ERR command_too_long");
    }
  }
}

}  // namespace

void setup() {
  Serial1.setTX(UART_TX_PIN);
  Serial1.setRX(UART_RX_PIN);
  Serial1.begin(BAUD_RATE);

  switchcontrollerpico_init();

  while (!TinyUSBDevice.mounted()) {
    delay(1);
  }

  switchcontrollerpico_reset();
  Serial1.println("serial-hid-bridge ready");
}

void loop() {
  readSerialCommands();
}
