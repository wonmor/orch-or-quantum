#!/bin/zsh
# usage: send.sh <draft.txt> <attachment.pdf>
# Sends via Mail.app from john@orchestrsim.com. First line "To:", second "Subject:", blank line, then body.
set -e
F="$1"; ATT="$2"
TO=$(sed -n '1s/^To: //p' "$F"); SUBJ=$(sed -n '2s/^Subject: //p' "$F"); BODY=$(tail -n +4 "$F")
osascript - "$TO" "$SUBJ" "$BODY" "$ATT" <<'APPLESCRIPT'
on run argv
  set theTo to item 1 of argv
  set theSubject to item 2 of argv
  set theBody to item 3 of argv
  set theAtt to POSIX file (item 4 of argv)
  tell application "Mail"
    set msg to make new outgoing message with properties {sender:"Wonmo (John) Seong <john@orchestrsim.com>", subject:theSubject, content:theBody & return & return, visible:false}
    tell msg
      make new to recipient at end of to recipients with properties {address:theTo}
      make new attachment with properties {file name:theAtt} at after the last paragraph
    end tell
    delay 2
    send msg
  end tell
  return "sent to " & theTo
end run
APPLESCRIPT
