import { useState, useEffect } from "react";
import { Button, Box, Spinner } from "@chakra-ui/react";
import { FaMicrophone, FaMicrophoneSlash } from "react-icons/fa";
import mixpanel from "mixpanel-browser";
import { MIXPANEL_TOKEN } from "../../main";
import { useHotkeys } from "react-hotkeys-hook";
import { HOTKEY_CONFIG } from "../../utils/constants";
import { getPlatformModifierKey } from "../../utils/platform";

export const Whisper = ({
  onConvertedText,
  apiKey,
}: {
  onConvertedText: (text: string) => void;
  apiKey: string | null;
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [hasRecordingSupport, setHasRecordingSupport] = useState(false);
  const [isDesktopDevice, setIsDesktopDevice] = useState(false);

  useEffect(() => {
    // Not inlined because of some TypeScript nonsense.
    if (navigator.mediaDevices && MediaRecorder) {
      setHasRecordingSupport(true);
    } else setHasRecordingSupport(false);

    setIsDesktopDevice(
      // https://stackoverflow.com/questions/11381673/detecting-a-mobile-browser
      !(
        window.navigator.userAgent?.toLowerCase()?.includes("mobi") ??
        window.innerWidth < 1024
      )
    );
  }, []);

  const onDataAvailable = (e: BlobEvent) => {
    const formData = new FormData();
    formData.append("file", e.data, "recording.webm");
    formData.append("model", "whisper-1");

    setIsTranscribing(true);

    fetch("https://api.openai.com/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => onConvertedText(data.text))
      .catch((err) => console.error("Error transcribing: ", err))
      .finally(() => setIsTranscribing(false));
  };

  const startRecording = async () => {
    setIsRecording(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const recorder = new MediaRecorder(stream);

      recorder.onstop = () => {
        stream.getTracks().forEach((track) => {
          track.stop();
        });

        if (stream.active) {
          stream.getTracks().forEach((track) => {
            stream.removeTrack(track);
          });
        }
      };

      recorder.addEventListener("dataavailable", onDataAvailable);
      recorder.start();

      setMediaRecorder(recorder);

      if (MIXPANEL_TOKEN) mixpanel.track("Started recording");
    } catch (error) {
      console.error("Error starting recorder: ", error);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) mediaRecorder.stop();

    setIsRecording(false);

    if (MIXPANEL_TOKEN) mixpanel.track("Stopped recording");
  };

  const modifierKey = getPlatformModifierKey();

  useHotkeys(
    `${modifierKey}+L`,
    () => (isRecording ? stopRecording() : startRecording()),
    HOTKEY_CONFIG
  );

  return (
    <>
      {hasRecordingSupport && isDesktopDevice && (
        <Box>
          <Button
            position="absolute"
            bottom={1}
            right={1}
            zIndex={10}
            variant="outline"
            border="0px"
            p={1}
            _hover={{ background: "none" }}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isTranscribing}
          >
            {isRecording ? (
              <FaMicrophoneSlash />
            ) : isTranscribing ? (
              <Spinner size="sm" />
            ) : (
              <FaMicrophone />
            )}
          </Button>
        </Box>
      )}
    </>
  );
};
