import QtQuick
import QtQuick.Layouts
import org.kde.plasma.plasmoid
import org.kde.plasma.components as PlasmaComponents
import org.kde.kirigami as Kirigami

// Running-job widget for dashboard.py (python3 dashboard.py, default http://127.0.0.1:8190).
PlasmoidItem {
    id: root

    property string url: "http://127.0.0.1:8190/api/state"
    property var st: null
    property string err: "connecting…"

    readonly property var job: (st && st.online && st.running.length) ? st.running[0] : null
    readonly property int queued: st && st.online ? st.pending.length : 0
    readonly property real frac: job && job.max ? job.value / job.max : 0
    readonly property string shortText: err ? "?" : !st.online ? "off" : job ? (job.max ? job.value + "/" + job.max : "…") : "idle"

    Plasmoid.icon: "applications-graphics"
    toolTipMainText: err ? "ComfyUI: " + err : !st.online ? "ComfyUI offline" : job ? "Job running" : "ComfyUI idle"
    toolTipSubText: job ? (job.node + " · " + fmt(job.elapsed) + (queued ? " · " + queued + " queued" : "")) : (queued ? queued + " queued" : "")

    function fmt(s) {
        return s >= 60 ? Math.floor(s / 60) + "m " + Math.round(s % 60) + "s" : Math.round(s) + "s"
    }

    function poll() {
        var x = new XMLHttpRequest()
        x.onreadystatechange = function() {
            if (x.readyState !== 4) return
            if (x.status === 200) {
                try { root.st = JSON.parse(x.responseText); root.err = "" }
                catch (e) { root.err = "bad data" }
            } else {
                root.st = null
                root.err = "dashboard not running"
            }
        }
        x.open("GET", url)
        x.send()
    }

    Timer {
        interval: 2000
        running: true
        repeat: true
        triggeredOnStart: true
        onTriggered: root.poll()
    }

    compactRepresentation: MouseArea {
        onClicked: root.expanded = !root.expanded
        implicitWidth: row.implicitWidth
        RowLayout {
            id: row
            anchors.fill: parent
            spacing: Kirigami.Units.smallSpacing
            Kirigami.Icon {
                source: "applications-graphics"
                Layout.preferredHeight: Kirigami.Units.iconSizes.smallMedium
                Layout.preferredWidth: Kirigami.Units.iconSizes.smallMedium
                opacity: root.job ? 1 : 0.5
            }
            PlasmaComponents.Label {
                text: root.shortText + (root.queued ? " +" + root.queued : "")
                color: root.err || (root.st && !root.st.online) ? Kirigami.Theme.negativeTextColor : Kirigami.Theme.textColor
            }
        }
    }

    fullRepresentation: ColumnLayout {
        Layout.minimumWidth: Kirigami.Units.gridUnit * 16
        Layout.minimumHeight: Kirigami.Units.gridUnit * 7
        spacing: Kirigami.Units.smallSpacing

        // header, like the system monitor widgets' title
        RowLayout {
            Layout.fillWidth: true
            Kirigami.Heading {
                text: "ComfyUI"
                level: 4
                Layout.fillWidth: true
            }
            PlasmaComponents.Label {
                text: "●"
                color: root.err || (root.st && !root.st.online) ? Kirigami.Theme.negativeTextColor
                     : root.job ? Kirigami.Theme.positiveTextColor : Kirigami.Theme.disabledTextColor
            }
        }
        Kirigami.Separator { Layout.fillWidth: true }

        PlasmaComponents.Label {
            visible: root.err !== "" || (root.st && !root.st.online)
            text: root.err || "ComfyUI is offline"
            color: Kirigami.Theme.negativeTextColor
            Layout.fillWidth: true
            wrapMode: Text.Wrap
        }
        PlasmaComponents.Label {
            visible: root.err === "" && root.st && root.st.online && !root.job
            text: "Idle" + (root.queued ? " · " + root.queued + " queued" : "")
            opacity: 0.7
        }
        ColumnLayout {
            visible: root.job !== null
            Layout.fillWidth: true
            PlasmaComponents.Label {
                text: root.job ? "Job running" : ""
                font.bold: true
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
            PlasmaComponents.Label {
                text: root.job ? (root.job.node || "starting…") + " · " + root.fmt(root.job.elapsed) : ""
                opacity: 0.7
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
            Rectangle {
                visible: root.job !== null
                Layout.fillWidth: true
                Layout.preferredHeight: Kirigami.Units.gridUnit * 1.4
                radius: height / 2
                color: Qt.rgba(Kirigami.Theme.textColor.r, Kirigami.Theme.textColor.g, Kirigami.Theme.textColor.b, 0.15)
                clip: true
                Rectangle {
                    width: parent.width * root.frac
                    height: parent.height
                    radius: parent.radius
                    color: Kirigami.Theme.positiveTextColor
                    Behavior on width { NumberAnimation { duration: 200 } }
                }
                PlasmaComponents.Label {
                    anchors.centerIn: parent
                    text: root.job && root.job.max ? root.job.value + "/" + root.job.max : "…"
                    font.bold: true
                }
            }
            PlasmaComponents.Label {
                visible: root.queued > 0
                text: root.queued + " queued"
                opacity: 0.7
            }
        }

        Item { Layout.fillHeight: true }  // pins the header + content to the top, so idle and running line up
    }
}
