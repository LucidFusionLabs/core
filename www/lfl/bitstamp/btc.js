function log(msg) { try { $("#log").prepend("<p>" + msg + "</p>"); } catch(e) {} }

function LessThan   (l, r) { return l < r ? true : false; }
function GreaterThan(l, r) { return r < l ? true : false; }

Array.prototype.last = function() { return this[this.length-1]; }
Array.prototype.binarySearch = function(needle, cmp, field) {
    var low = 0, high = this.length, ind, missing;
    while (low <= high) {
        ind = Math.floor((low + high) / 2);
        missing = this[ind] == undefined;
        if (!missing &&     this[ind][field] == needle)  return ind;
        if (!missing && cmp(this[ind][field],   needle)) low  = ind + 1;
        else                                             high = ind - 1;
    }
    return low;
};

function FormatPrice (price,  digits) { return parseFloat(price ).toFixed(digits || 3); }
function FormatVolume(volume, digits) { return parseFloat(volume).toFixed(digits || 4); }

var TYPE_CANDLE=1, TYPE_LINE=2, TYPE_DEPTH=3;
var ChartType = TYPE_DEPTH;

var SOURCE_ALL=1, SOURCE_BUYS=2, SOURCE_SELLS=3;
var ChartSource = SOURCE_ALL;

function GetTicks() {
    if      (ChartSource == SOURCE_ALL)   return Ticks;
    else if (ChartSource == SOURCE_BUYS)  return BidTicks;
    else if (ChartSource == SOURCE_SELLS) return AskTicks;
    else return null;
}

var bg_color = $("body").css("background-color");
var border_color = "#555555";
var green_fill_color = "#00aa00", green_outline_color = "#00ff00";
var red_fill_color   = "#aa0000", red_outline_color   = "#ff0000";

var chart = document.getElementById("chart");
var canvas = chart.getContext("2d");

var border = 5;
var width = chart.width;
var height = chart.height;
var label_width = width * .09;
var bar_width = width * .008;
var bar_padding = 2;
var volume_frame_height = height * .3;
var between_chart_spacing = 10;
var bottom_label_height = 40;

var price_frame_left = border;
var price_frame_right = width - border*2 - label_width;

var price_frame_top = border;
var price_frame_bottom = height - border*2 - volume_frame_height - between_chart_spacing;

var volume_frame_top = height - border - volume_frame_height;
var volume_frame_bottom = height - border;

var depth_frame_left  =         border*2 + label_width;
var depth_frame_right = width - border*2 - label_width;

var depth_frame_top = border;
var depth_frame_bottom = height - border - bottom_label_height;

var price_chart_left = price_frame_left + border;
var price_chart_right = price_frame_right - border;
var price_chart_width = price_chart_right - price_chart_left;

var price_chart_top = price_frame_top + border;
var price_chart_bottom = price_frame_bottom - border;
var price_chart_height = price_chart_bottom - price_chart_top;

var volume_chart_top = volume_frame_top + border;
var volume_chart_bottom = volume_frame_bottom - border;
var volume_chart_height = volume_chart_bottom - volume_chart_top;

var depth_chart_left = depth_frame_left + border;
var depth_chart_right = depth_frame_right - border;
var depth_chart_width = depth_chart_right - depth_chart_left;

var depth_chart_top = depth_frame_top + border;
var depth_chart_bottom = depth_frame_bottom - border;
var depth_chart_height = depth_chart_bottom - depth_chart_top;

var right_label_right = width - border;
var right_label_left = width - border - label_width;

var left_label_left = border;
var left_label_right = border + label_width;

var price_min, price_max, price_range, volume_max;
var depth_min, depth_max, depth_mid, depth_range, depth_volume_max;

function DrawChart() {
    canvas.globalAlpha=0.8;
    canvas.fillStyle = bg_color;

    canvas.clearRect(0, 0, width, height);
    canvas.fillRect(0, 0, width, height);

    var ticks = GetTicks();
    if (!ticks) return;

    if (ticks.length) window.document.title = "Bitstamp Live " + FormatPrice(ticks.last()[CLOSE], 2);

    if (ChartType == TYPE_CANDLE && ticks.length) {
        FindPriceVolumeRange(ticks);
        DrawPriceChartBorder();
        DrawPriceChartLabels();
        DrawCandlestickChart(ticks);
        DrawVolumeChart(ticks);
    } else if (ChartType == TYPE_LINE && ticks.length) {
        FindPriceVolumeRange(ticks);
        DrawPriceChartBorder();
        DrawPriceChartLabels();
        DrawLineChart(ticks);
        DrawVolumeChart(ticks);
    } else if (ChartType == TYPE_DEPTH && Asks.length && Bids.length) {
        FindDepthRange();
        DrawDepthChartBorder();
        DrawDepthChartLabels();
        DrawDepthChart();
    }
}

function FindPriceVolumeRange(ticks) {
    volume_max = 0;
    price_max = 0;
    price_min = Number.MAX_VALUE;
    for (var i = ticks.length-1; i >= 0; i--) {
        var x = price_chart_right - (ticks.length - i) * (bar_width + bar_padding*2);
        if (x < price_chart_left) break;

        var tick = ticks[i];
        if (tick[HIGH]   > price_max)  price_max  = tick[HIGH];
        if (tick[LOW]    < price_min)  price_min  = tick[LOW];
        if (tick[VOLUME] > volume_max) volume_max = tick[VOLUME];
    }
    price_range = price_max - price_min;
    price_min -= price_range * .1;
    price_max += price_range * .1;
    price_range = price_max - price_min;
}

function FindDepthRange() {
    depth_mid = Bids[0].price + (Asks[0].price - Bids[0].price) / 2;
    depth_range = depth_mid * .05;
    depth_min = depth_mid - depth_range;
    depth_max = depth_mid + depth_range;

    var bids_volume = 0, asks_volume = 0;
    for (var i = 0; i < Bids.length && Bids[i].price > depth_min; i++) bids_volume += Bids[i].volume;
    for (var i = 0; i < Asks.length && Asks[i].price < depth_max; i++) asks_volume += Asks[i].volume;
    depth_volume_max = Math.max(bids_volume, asks_volume);
}

function DrawPriceChartBorder() {
    canvas.strokeStyle = border_color;
    canvas.lineWidth = 1;

    canvas.beginPath();
    canvas.strokeRect(price_frame_left, price_frame_top,
                      price_frame_right - price_frame_left, price_frame_bottom - price_frame_top);
    canvas.strokeRect(price_frame_left, volume_frame_top,
                      price_frame_right - price_frame_left, volume_frame_bottom - volume_frame_top);
    canvas.stroke();
}

function DrawDepthChartBorder() {
    canvas.strokeStyle = border_color;
    canvas.lineWidth = 1;

    canvas.beginPath();
    canvas.strokeRect(depth_frame_left, depth_frame_top,
                      depth_frame_right - depth_frame_left, depth_frame_bottom - depth_frame_top);
    canvas.stroke();
}

function DrawPriceChartLabels() {
    canvas.fillStyle = "#999999";
    canvas.textBaseline = "middle";
    canvas.font = "small-caps bold 10px arial";
    var text_width = canvas.measureText(FormatPrice(price_max, 2)).width;

    for (var i = 0; i <= 10; i++) {
        var price = price_min + i/10 * price_range;
        var y = price_chart_bottom - i/10 * price_chart_height;
        canvas.fillText(FormatPrice(price, 2), right_label_right - text_width, y);
    }

    for (var i = 0; i <= 4; i++) {
        var volume = i/4 * volume_max;
        var y = volume_chart_bottom - i/4 * volume_chart_height;
        canvas.fillText(FormatVolume(volume, 2), right_label_right - text_width, y);
    }
}

function DrawDepthChartLabels() {
    canvas.fillStyle = "#999999";
    canvas.textBaseline = "middle";
    canvas.font = "small-caps bold 10px arial";
    var text_width = canvas.measureText(FormatPrice(depth_max, 2)).width;

    for (var i = 0; i <= 10; i++) {
        var volume = i/10 * depth_volume_max;
        var y = depth_chart_bottom - i/10 * depth_chart_height;
        canvas.fillText(FormatVolume(volume, 2), right_label_right - text_width, y);
        canvas.fillText(FormatVolume(volume, 2), left_label_left, y);
    }

    canvas.textBaseline = "top";
    for (var i = 0; i <= 10; i++) {
        var price = depth_mid + (-1 + i/5) * depth_range;
        var text = FormatPrice(price, 2);
        var text_width = canvas.measureText(text).width;
        var x = depth_chart_left - text_width/2 + i/10 * depth_chart_width;
        canvas.fillText(text, x, depth_chart_bottom + 15);
    }
}

function DrawCandlestickChart(ticks) {
    var green_outline = [], green_filled = [], red_outline = [], red_filled = [];
    for (var i = ticks.length-1; i > 0; i--) {
        var x = price_chart_right - (ticks.length - i) * (bar_width + bar_padding*2);
        if (x < price_chart_left) break;

        var tick = ticks[i], prev_tick = ticks[i-1];
        var o = price_chart_bottom - (tick[OPEN]  - price_min) / price_range * price_chart_height;
        var h = price_chart_bottom - (tick[HIGH]  - price_min) / price_range * price_chart_height;
        var l = price_chart_bottom - (tick[LOW]   - price_min) / price_range * price_chart_height;
        var c = price_chart_bottom - (tick[CLOSE] - price_min) / price_range * price_chart_height;

        if (tick[CLOSE] > prev_tick[CLOSE] ||
            (tick[CLOSE] == prev_tick[CLOSE] && tick[OPEN] >= prev_tick[OPEN])) {
            if (tick[CLOSE] >= tick[OPEN]) green_outline.push([x, o, h, l, c]);
            else                           green_filled .push([x, o, h, l, c]);
        } else {
            if (tick[CLOSE] >= tick[OPEN]) red_outline  .push([x, o, h, l, c]);
            else                           red_filled   .push([x, o, h, l, c]);
        }
    }

    canvas.strokeStyle = green_outline_color;
    canvas.fillStyle   = green_fill_color;
    canvas.beginPath();
    for (var i = 0; i < green_outline.length; i++) {
        var t = green_outline[i];
        DrawCandlestickLines(canvas, t[0], t[1], t[2], t[3], t[4]);
    }
    for (var i = 0; i < green_filled.length; i++) {
        var t = green_filled[i];
        DrawCandlestickBars (canvas, t[0], t[1], t[2], t[3], t[4]);
        DrawCandlestickLines(canvas, t[0], t[1], t[2], t[3], t[4]);
    }
    canvas.stroke();

    canvas.strokeStyle = red_outline_color;
    canvas.fillStyle   = red_fill_color;
    canvas.beginPath();
    for (var i = 0; i < red_outline.length; i++) {
        var t = red_outline[i];
        DrawCandlestickLines(canvas, t[0], t[1], t[2], t[3], t[4]);
    }
    for (var i = 0; i < red_filled.length; i++) {
        var t = red_filled[i];
        DrawCandlestickBars (canvas, t[0], t[1], t[2], t[3], t[4]);
        DrawCandlestickLines(canvas, t[0], t[1], t[2], t[3], t[4]);
    }
    canvas.stroke();
}

function DrawLineChart(ticks) {
    var green = [], red = [];
    for (var i = ticks.length-1; i > 0; i--) {
        var x  = price_chart_right - (ticks.length - i    ) * (bar_width + bar_padding*2);
        var px = price_chart_right - (ticks.length - i + 1) * (bar_width + bar_padding*2);
        if (x < price_chart_left) break;

        var tick = ticks[i], prev_tick = ticks[i-1];
        var c  = price_chart_bottom - (     tick[CLOSE] - price_min) / price_range * price_chart_height;
        var pc = price_chart_bottom - (prev_tick[CLOSE] - price_min) / price_range * price_chart_height;

        if (tick[CLOSE] > prev_tick[CLOSE] ||
            (tick[CLOSE] == prev_tick[CLOSE] && tick[OPEN] >= prev_tick[OPEN])) green.push([px, pc, x, c]);
        else                                                                      red.push([px, pc, x, c]);
    }

    canvas.lineWidth   = 3;
    canvas.strokeStyle = green_outline_color;
    canvas.fillStyle   = green_fill_color;
    canvas.beginPath();
    for (var i = 0; i < green.length; i++) {
        var segment = green[i];
        DrawLineChartSegment(canvas, segment[0], segment[1], segment[2], segment[3]);
    }
    canvas.stroke();

    canvas.strokeStyle = red_outline_color;
    canvas.fillStyle   = red_fill_color;
    canvas.beginPath();
    for (var i = 0; i < red.length; i++) {
        var segment = red[i];
        DrawLineChartSegment(canvas, segment[0], segment[1], segment[2], segment[3]);
    }
    canvas.stroke();
}

function DrawVolumeChart(ticks) {
    var green = [], red = [];
    for (var i = ticks.length-1; i > 0; i--) {
        var x = price_chart_right - (ticks.length - i) * (bar_width + bar_padding*2);
        if (x < price_chart_left) break;

        var tick = ticks[i], prev_tick = ticks[i-1];
        var y = volume_chart_bottom - tick[VOLUME] / volume_max * volume_chart_height;

        if (tick[CLOSE] >= prev_tick[CLOSE]) green.push([x, y]);
        else                                   red.push([x, y]);
    }

    canvas.lineWidth   = 1;
    canvas.strokeStyle = green_outline_color;
    canvas.fillStyle   = green_fill_color;
    canvas.beginPath();
    for (var i = 0; i < green.length; i++) DrawVolumeBars(canvas, green[i][0], green[i][1]);
    canvas.stroke();

    canvas.strokeStyle = red_outline_color;
    canvas.fillStyle   = red_fill_color;
    canvas.beginPath();
    for (var i = 0; i < red.length; i++) DrawVolumeBars(canvas, red[i][0], red[i][1]);
    canvas.stroke();
}

function DrawDepthChart() {
    var grd = canvas.createLinearGradient(depth_chart_left + depth_chart_width/4, depth_chart_top,
                                          depth_chart_left + depth_chart_width/4, depth_chart_bottom);
    grd.addColorStop(0, red_fill_color);
    grd.addColorStop(1, "#000000");

    canvas.lineWidth   = 1;
    canvas.strokeStyle = red_outline_color;
    canvas.fillStyle   = grd;
    canvas.beginPath();
    DrawHalfDepthChart(canvas, Bids, depth_min, -1);
    canvas.fill();
    canvas.stroke();

    grd = canvas.createLinearGradient(depth_chart_right - depth_chart_width/4, depth_chart_top,
                                      depth_chart_right - depth_chart_width/4, depth_chart_bottom);
    grd.addColorStop(0, green_fill_color);
    grd.addColorStop(1, "#000000");

    canvas.strokeStyle = green_outline_color;
    canvas.fillStyle   = grd;
    canvas.beginPath();
    DrawHalfDepthChart(canvas, Asks, depth_max, 1);
    canvas.fill();
    canvas.stroke();
}

function DrawHalfDepthChart(canvas, orders, limit_price, mult) {
    var x, y, total_volume = 0;
    var half_depth_chart_width = depth_chart_width/2;
    var depth_chart_mid = depth_chart_left + half_depth_chart_width;

    canvas.moveTo(depth_chart_mid, depth_chart_bottom);
    for (var i = 0; i < orders.length && mult * (limit_price - orders[i].price) > 0; i++) {
        total_volume += orders[i].volume;

        y = depth_chart_bottom - total_volume / depth_volume_max * depth_chart_height;
        x = depth_chart_mid + mult *
            Math.abs(depth_mid - orders[i].price) / depth_range * half_depth_chart_width;

        canvas.lineTo(x, y);
    }

    var end_x = depth_chart_mid + half_depth_chart_width * mult;
    canvas.lineTo(end_x, y);
    canvas.lineTo(end_x, depth_chart_bottom);
}

function DrawVolumeBars(canvas, x, y) {
    canvas.fillRect  (x + bar_padding, y, bar_width, volume_chart_bottom - y);
    canvas.strokeRect(x + bar_padding, y, bar_width, volume_chart_bottom - y);
}

function DrawCandlestickBars(canvas, x, o, h, l, c) {
    var top = Math.min(o, c), bottom = Math.max(o, c);
    canvas.fillRect(x + bar_padding, top, bar_width, bottom - top);
}

function DrawCandlestickLines(canvas, x, o, h, l, c) {
    var top = Math.min(o, c), bottom = Math.max(o, c);
    canvas.strokeRect(x + bar_padding, top, bar_width, bottom - top);
    canvas.moveTo(x + bar_padding + bar_width/2, bottom);
    canvas.lineTo(x + bar_padding + bar_width/2, l);
    canvas.moveTo(x + bar_padding + bar_width/2, top);
    canvas.lineTo(x + bar_padding + bar_width/2, h);
}

function DrawLineChartSegment(canvas, sx, sy, tx, ty) {
    canvas.moveTo(sx + bar_padding + bar_width, sy);
    canvas.lineTo(tx + bar_padding + bar_width, ty);
}

var crosshair_x = 0, crosshair_y = 0, crosshair_top = 0, crosshair_height = 0, crosshair_chart = "";
var chart_z1 = document.getElementById("chart_z1");
var canvas_z1 = chart_z1.getContext("2d");

$("#chart_z1").on({
    mouseout:  function(e) { ClearPreviousCrosshairs(); },
    mousemove: function(e) {
        ClearPreviousCrosshairs();

        var ticks = GetTicks();
        if (!ticks || (ChartType != TYPE_CANDLE && ChartType != TYPE_LINE)) return;

        if (e.offsetX < price_chart_left || e.offsetX > price_chart_right) return;

        if (e.offsetY >= price_chart_top && e.offsetY <= price_chart_bottom) {
            crosshair_chart  = "price";
            crosshair_top    = price_chart_top;
            crosshair_height = price_chart_height;
        }
        else if (e.offsetY >= volume_chart_top && e.offsetY <= volume_chart_bottom) {
            crosshair_chart  = "volume";
            crosshair_top    = volume_chart_top;
            crosshair_height = volume_chart_height;
        }
        else return;

        crosshair_x = e.offsetX;
        crosshair_y = e.offsetY;

        var tick_ind = ticks.length-1 - Math.floor((price_chart_right - crosshair_x) / (bar_width + bar_padding*2));
        if (tick_ind >= 0) {
            canvas_z1.fillStyle = "#ffffff";
            canvas_z1.textBaseline = "top";
            crosshair_x = price_chart_right - (ticks.length - tick_ind - .5) * (bar_width + bar_padding*2);

            var tick = ticks[tick_ind];
            var text = new Date(tick[TIME]).toLocaleString() + ", ";
            if (crosshair_chart == "price") {
                text += "O: " + FormatPrice(tick[OPEN], 2) + ", H: " + FormatPrice(tick[HIGH],  2) +
                      ", L: " + FormatPrice(tick[LOW],  2) + ", C: " + FormatPrice(tick[CLOSE], 2);
                canvas_z1.fillText(text, price_chart_left, crosshair_top);
            }
            else if (crosshair_chart == "volume") {
                text += "V: " + FormatVolume(tick[VOLUME], 2);
                canvas_z1.fillText(text, price_chart_left, crosshair_top);
            }
        }

        canvas_z1.strokeStyle = "#ffffff";
        canvas_z1.beginPath();
        canvas_z1.moveTo(crosshair_x, crosshair_top);
        canvas_z1.lineTo(crosshair_x, crosshair_top + crosshair_height);
        canvas_z1.moveTo(price_chart_left,  crosshair_y);
        canvas_z1.lineTo(price_chart_right, crosshair_y);
        canvas_z1.stroke();
    }
});

$("#order_book_controls a:contains('more')").click(function() {
    DisplayRows++;
    PushOrderbookRowHTML(Asks, "ask");
    PushOrderbookRowHTML(Bids, "bid");
});

$("#order_book_controls a:contains('less')").click(function() {
    if (!DisplayRows) return;
    DisplayRows--;
    PopOrderbookRowHTML(Asks, "ask");
    PopOrderbookRowHTML(Bids, "bid");
});

$("#chart_controls a:contains('settings')").click(function() {
    $("#chart_settings_dialog").dialog("open");
});

$("#chart_settings_dialog").dialog({ autoOpen: false, width: 500 });

$('#chart_type').change(function(e) {
    ChartType = parseInt(e.target.value);
    DrawChart();
});

$('#chart_source').change(function(e) {
    ChartSource = parseInt(e.target.value);
    DrawChart();
});

function ClearPreviousCrosshairs() {
    canvas_z1.clearRect(price_chart_left, crosshair_top, price_chart_width, 15);
    canvas_z1.clearRect(crosshair_x-1.5, crosshair_top, 3, crosshair_height);
    canvas_z1.clearRect(price_chart_left, crosshair_y-1.5, price_chart_width, 3);
}

var HandleMessages = true;
var CompressEmptyTicks = true;

var ServerDomain = "127.0.0.1";
var ServerPort = 8090;

var TradeChannel = { name: "trade", init: false, first_time: 0, last_time: 0, buffer: [], handler: HandleTrade };
var DepthChannel = { name: "depth", init: false, first_time: 0, last_time: 0, buffer: [], handler: HandleDepth };

var Ticks = [], AskTicks = [], BidTicks = [];
var TIME=0, OPEN=1, HIGH=2, LOW=3, CLOSE=4, VOLUME=5;

var MinVolume = 0, DisplayRows = 10;
var Bids = [], Asks = [];

var pusher = null;

function PusherSubscribe(channel_name, bind_name, connect_cb, data_cb) {
    var channel = pusher.subscribe(channel_name);
    channel.bind(bind_name, data_cb);
    connect_cb();
}

function OpenBitstamp() {
    pusher = new Pusher('de504dc5763aeef9ff52');
    PusherSubscribe('live_trades', 'trade', FetchArchivedTicks, HandleTrade);
    PusherSubscribe('diff_order_book', 'data', FetchOrderbookState, HandleDepth);
}

function FetchArchivedTicks() {
    $.ajax({ url: "http://" + ServerDomain + ":8090/bitstamp_archive/tick/M1",
             type: "GET",
             dataType: "json",
             success: function(msg) { HandleArchivedTicks(msg); },
             error:   function(msg) { HandleArchivedTicks(null); }
           });
}

function FetchOrderbookState() {
    $.ajax({ url: "http://" + ServerDomain + ":8090/bitstamp_archive/orderbook",
             type: "GET",
             dataType: "json",
             success: function(msg) { HandleOrderbookState(msg); },
             error:   function(msg) { HandleOrderbookState(null); }
           });
}

function HandleArchivedTicks(msg) {
    if (!msg) console.log("HandleArchivedTicks: Connection error");
    else {
        Ticks    = msg.ticks;
        AskTicks = msg.asks;
        BidTicks = msg.bids;
    }

    for (var i = 0; i < TradeChannel.buffer.length; i++) TradeChannel.handler(TradeChannel.buffer[i]);
    TradeChannel.buffer = [];
    TradeChannel.init = true;

    DrawChart();
}

function HandleOrderbookState(msg) {
    if (!msg) console.log("HandleOrderbookState: Connection error");
    else {
        for (var i = 0; i < msg.asks.length; i++)
            if (msg.asks[i].volume > MinVolume) Asks.push(msg.asks[i]);
        for (var i = 0; i < msg.bids.length; i++)
            if (msg.bids[i].volume > MinVolume) Bids.push(msg.bids[i]);

        for (var i = 0; i < Asks.length && i < DisplayRows; i++)
            $("#order_ask tr:nth-child(" + (i+1) + ")").replaceWith(GetOrderbookRowHTML(Asks[i]));
        for (var i = 0; i < Bids.length && i < DisplayRows; i++)
            $("#order_bid tr:nth-child(" + (i+1) + ")").replaceWith(GetOrderbookRowHTML(Bids[i]));

        AlternateOrderbookRowClass("ask");
        AlternateOrderbookRowClass("bid");
    }
    if (ChartType == TYPE_DEPTH) DrawChart();

    for (var i = 0; i < DepthChannel.buffer.length; i++) DepthChannel.handler(DepthChannel.buffer[i]);
    DepthChannel.buffer = [];
    DepthChannel.init = true;
}

function HandleTrade(msg) {
    var trade_time = (new Date).getTime();
    var trade_price = parseFloat(msg.price);
    var trade_amount = parseFloat(msg.amount);
    AddTrade(Ticks, trade_time, trade_price, trade_amount);
    DrawChart();
}

function AddTrade(ticks, trade_time, trade_price, trade_amount) {
    if (ticks.length == 0) ticks.push([ trade_time, 0, 0, 0, 0, 0 ]);

    var last_tick = ticks.last();
    var last_tick_min = Math.floor(last_tick[TIME] / 60000);
    var cur_tick_min = Math.floor(trade_time / 60000);
    var mins_since_last_tick = cur_tick_min - last_tick_min;
    var start_i = mins_since_last_tick && CompressEmptyTicks ? mins_since_last_tick-1 : 0;
    for (var i = start_i; i < mins_since_last_tick; i++) {
        var min = last_tick_min + i + 1;
        var v = i != mins_since_last_tick-1 ? last_tick[CLOSE] : 0;
        ticks.push([ min * 60000, v, v, v, v, 0 ]);
    }
    while (ticks.length > 180) ticks.shift();

    var cur_tick = ticks.last();
    if (cur_tick[OPEN] == 0) {
        cur_tick[OPEN] = trade_price;
        cur_tick[LOW]  = trade_price;
    }

    if (cur_tick[HIGH] < trade_price) cur_tick[HIGH] = trade_price;
    if (cur_tick[LOW]  > trade_price) cur_tick[LOW]  = trade_price;

    cur_tick[CLOSE] = trade_price;
    cur_tick[VOLUME] += trade_amount;
}

function HandleDepth(msg) {
    for (var i = 0; i < msg.asks.length; i++) {
        var ask = msg.asks[i];
        HandleDepthUpdate({ type: "ask", price: parseFloat(ask[0]), volume: parseFloat(ask[1]),
                          timeout_price: undefined, timeout_volume: undefined });
    }
    for (var i = 0; i < msg.bids.length; i++) {
        var bid = msg.bids[i];
        HandleDepthUpdate({ type: "bid", price: parseFloat(bid[0]), volume: parseFloat(bid[1]),
                          timeout_price: undefined, timeout_volume: undefined });
    }
}

function HandleDepthUpdate(order) {
    var orders  = order.type == "bid" ? Bids        : Asks;
    var cmp     = order.type == "bid" ? GreaterThan : LessThan;

    var orders_ind = orders.binarySearch(order.price, cmp, 'price');
    var out = "#order_" + order.type + " tr:nth-child(" + (orders_ind + 1) + ")";

    if (orders[orders_ind] == undefined || orders[orders_ind].price != order.price) {
        if (order.volume > MinVolume) {
            orders.splice(orders_ind, 0, order);

            if (orders_ind < DisplayRows) {
                $(out).before(GetOrderbookRowHTML(order, "insert"));

                var price_ind  = order.type == "bid" ? 2 : 1;
                var volume_ind = order.type == "bid" ? 1 : 2;
                FadeOrderbookCell(order, "price",  $(out + " td:nth-child(" + price_ind  + ")"));
                FadeOrderbookCell(order, "volume", $(out + " td:nth-child(" + volume_ind + ")"));

                PopOrderbookRowHTML(orders, order.type);
            }
        }
    } else {
        orders[orders_ind].volume = order.volume;
        order = orders[orders_ind];
        clearTimeout(order.timeout_volume);

        if (orders_ind < DisplayRows) {
            $(out).replaceWith(GetOrderbookRowHTML(order, "change"));

            var volume_ind = order.type == "bid" ? 1 : 2;
            FadeOrderbookCell(order, "volume", $(out + " td:nth-child(" + volume_ind + ")"));
        }
        else if (order.volume <= MinVolume) {
            clearTimeout(order.timeout_price);
            orders.splice(orders_ind, 1);
        }
    }

    if (ChartType == TYPE_DEPTH) DrawChart();
}

function FadeOrderbookCell(order, timeout_field, out) {
    order["timeout_" + timeout_field] = setTimeout(function() {
        out.removeClass("order_changed");
        order["timeout_" + timeout_field] = undefined;
        if (timeout_field == "volume" && order.volume <= MinVolume) DelOrderbookRow(order.type, order.price);
    }, 1000);
}

function DelOrderbookRow(type, price) {
    var orders = type == "bid" ? Bids        : Asks;
    var cmp    = type == "bid" ? GreaterThan : LessThan;

    var orders_ind = orders.binarySearch(price, cmp, 'price');
    var out = "#order_" + type + " tr:nth-child(" + (orders_ind + 1) + ")";

    if (orders[orders_ind] == undefined || orders[orders_ind].price != price)
        throw("missing order " + type + " " + price);

    clearTimeout(orders[orders_ind].timeout_price);
    orders.splice(orders_ind, 1);

    if (orders_ind < DisplayRows) {
        $(out).remove();
        PushOrderbookRowHTML(orders, type);
    }
}

function ShiftOrderbookRow(orders, type) {
    $("#order_" + type + " tr:first-child").remove();
    clearTimeout(orders[0].timeout_price);
    clearTimeout(orders[0].timeout_volume);
    orders.shift();
    PushOrderbookRowHTML(orders, type);
}

function PushOrderbookRowHTML(orders, type) {
    if (orders.length >= DisplayRows) {
        $("#order_" + type).append(GetOrderbookRowHTML(orders[DisplayRows-1], type));
    } else { 
        AddBlankOrderbookRowHTML(type);
    }
    AlternateOrderbookRowClass(type);
}

function PopOrderbookRowHTML(orders, type) {
    $("#order_" + type + " tr:last-child").remove();

    if (orders.length > DisplayRows) {
        clearTimeout(orders[DisplayRows].timeout_price);
        clearTimeout(orders[DisplayRows].timeout_volume);
        if (orders[DisplayRows].volume <= MinVolume) orders.splice(DisplayRows, 1);
    }

    AlternateOrderbookRowClass(type);
}

function GetOrderbookRowHTML(order, op) {
    var price  = FormatPrice(order.price);
    var volume = FormatVolume(order.volume);

    var v1 = order.type == "bid" ? volume : price;
    var v2 = order.type == "bid" ? price  : volume;

    var v1_class = (order.type == "bid" && op == "change") || op == "insert" ? "order_changed" : "";
    var v2_class = (order.type != "bid" && op == "change") || op == "insert" ? "order_changed" : "";

    var html = '<tr class="order">';
    html += '<td class="' + v1_class + '">' + v1 + "</td>";
    html += '<td class="' + v2_class + '">' + v2 + "</td>";
    html += "</tr>";
    return html;
}

function AddBlankOrderbookRowHTML(type) {
    var out = $("#order_" + type);
    var len = $("#order_" + type + " tr").length;
    var add_rows = DisplayRows - len;
    for (i = 0; i < add_rows; i++) out.append("<tr><td>&nbsp;</td><td>&nbsp;</td></tr>");
}

function AlternateOrderbookRowClass(type) {
    $("#order_" + type + " tr:odd").     addClass("order_alt");
    $("#order_" + type + " tr:even ").removeClass("order_alt");
}

AddBlankOrderbookRowHTML("bid");
AddBlankOrderbookRowHTML("ask");
OpenBitstamp();
