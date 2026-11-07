import QtQuick
import QtQuick.Layouts
import org.kde.plasma.plasmoid
import org.kde.plasma.components as PlasmaComponents
import org.kde.kirigami as Kirigami

// RunPod animate-pod widget for dashboard.py (python3 dashboard.py, default http://127.0.0.1:8190): the pod (down / booting / up,
// GPU, $/hr, the account balance, the last cold start) and, while it is up, its running job like the other two widgets.
PlasmoidItem {
    id: root

    property string url: "http://127.0.0.1:8190/api/state?target=runpod"
    property var st: null
    property string err: "connecting…"

    readonly property var job: (st && st.online && st.running.length) ? st.running[0] : null
    readonly property int queued: st && st.online ? st.pending.length : 0
    readonly property real frac: job && job.max ? job.value / job.max : 0
    readonly property var pod: st && st.pod ? st.pod : ({})
    readonly property bool podUp: pod.id !== undefined
    readonly property string shortText: err ? "?" : !podUp ? "off" : !st.online ? "boot" : job ? (job.max ? job.value + "/" + job.max : "…") : "idle"
    readonly property string podText: !podUp ? "No pod running" + (pod.last_run ? " · last run $" + pod.last_run.cost.toFixed(2) + " (" + Math.round(pod.last_run.minutes) + " min)" : "") : (pod.gpu || "").replace(/^NVIDIA (GeForce )?/, "")
        + (pod.cost_per_hr != null ? " · $" + pod.cost_per_hr.toFixed(2) + "/hr" : "") + (pod.uptime != null ? " · up " + fmt(pod.uptime) : "")
        + (pod.session_cost != null ? " · session $" + pod.session_cost.toFixed(2) : "")
    readonly property string moneyText: st && st.balance ? "Balance $" + st.balance.balance.toFixed(2) + " · spending $" + st.balance.spend_per_hr.toFixed(2) + "/hr" : ""
    readonly property var cold: pod.last_start || null
    readonly property string coldText: cold ? "Last cold start: ready " + fmt(cold.ready_s) + (cold.first_clip_s != null ? " · first clip " + fmt(cold.first_clip_s) : "")
        + (cold.warm_clip_s != null ? " · next " + fmt(cold.warm_clip_s) : "") : ""

    Plasmoid.icon: "network-server"
    toolTipMainText: err ? "RunPod: " + err : !podUp ? "RunPod pod down" : !st.online ? "RunPod pod booting" : job ? "Job running" : "RunPod pod idle"
    toolTipSubText: (podUp ? podText + "\n" : "") + moneyText + (job ? "\n" + job.node + " · " + fmt(job.elapsed) + (queued ? " · " + queued + " queued" : "") : (queued ? "\n" + queued + " queued" : ""))

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
                source: "network-server"
                Layout.preferredHeight: Kirigami.Units.iconSizes.smallMedium
                Layout.preferredWidth: Kirigami.Units.iconSizes.smallMedium
                opacity: root.podUp ? 1 : 0.5
            }
            PlasmaComponents.Label {
                text: root.shortText + (root.queued ? " +" + root.queued : "")
                color: root.err ? Kirigami.Theme.negativeTextColor : root.podUp ? Kirigami.Theme.neutralTextColor : Kirigami.Theme.disabledTextColor
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
                text: "RunPod"
                level: 4
                Layout.fillWidth: true
            }
            PlasmaComponents.Label {
                text: "●"
                color: root.err ? Kirigami.Theme.negativeTextColor : root.job ? Kirigami.Theme.positiveTextColor
                     : root.podUp ? Kirigami.Theme.neutralTextColor : Kirigami.Theme.disabledTextColor
            }
        }
        Kirigami.Separator { Layout.fillWidth: true }

        PlasmaComponents.Label {
            visible: root.err !== ""
            text: root.err
            color: Kirigami.Theme.negativeTextColor
            Layout.fillWidth: true
            wrapMode: Text.Wrap
        }
        // the pod (it bills while up) and the account: $/hr, balance, how long the last cold start took
        PlasmaComponents.Label {
            visible: root.err === ""
            text: root.podText + (root.podUp && root.st && !root.st.online ? " · booting…" : "")
            font.bold: root.podUp
            elide: Text.ElideRight
            Layout.fillWidth: true
        }
        PlasmaComponents.Label {
            visible: root.moneyText !== ""
            text: root.moneyText
            opacity: 0.7
            elide: Text.ElideRight
            Layout.fillWidth: true
        }
        PlasmaComponents.Label {
            visible: root.coldText !== ""
            text: root.coldText
            opacity: 0.7
            elide: Text.ElideRight
            Layout.fillWidth: true
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
