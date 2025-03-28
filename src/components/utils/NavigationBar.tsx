import { Button, Box, Text } from "@chakra-ui/react";

import { Row } from "../../utils/chakra";

export function NavigationBar({
  onOpenSettingsModal,
  onToggleAnswerFilter,
  showAnswerPathOnly,
}: {
  onOpenSettingsModal: () => void;
  onToggleAnswerFilter: () => void;
  showAnswerPathOnly: boolean;
}) {
  return (
    <Row
      mainAxisAlignment="flex-start"
      crossAxisAlignment="center"
      height="100%"
      width="auto"
    >
      <Text whiteSpace="nowrap">
        <b>Cube Solver</b>
      </Text>

      <Box mx="20px" height="100%" width="1px" bg="#EEEEEE" />

      <Button
        variant="ghost"
        height="80%"
        px="5px"
        ml="11px"
        onClick={onOpenSettingsModal}
      >
        Settings
      </Button>
      <Button
        variant="ghost"
        height="80%"
        px="5px"
        ml="11px"
        onClick={onToggleAnswerFilter}
      >
        {showAnswerPathOnly ? "Hide Answer Path" : "Show Answer Path"}
      </Button>
    </Row>
  );
}
